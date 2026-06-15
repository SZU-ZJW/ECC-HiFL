"""Trainable ECR reward-model scaffold.

Dataset construction runs **now, offline**: for each instance we parse the BoN pool, label each
candidate by whether it overlaps the gold (recall > 0), and emit:
- ``sft.jsonl``      — instruction-tuning examples ``{instruction, input, output:{"selected_index"}}``
                       where the target is the highest-recall (positive) candidate.
- ``pairwise.jsonl`` — ``{"chosen": <text>, "rejected": <text>}`` ranking pairs (positive vs negative).

Training (``train`` subcommand) is **opt-in / GPU-gated**: it lazily imports torch/transformers
and runs a Bradley-Terry pairwise objective on a sequence-classification reward head. Without
those deps it prints how to install/run.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from ..config import ECRConfig
from ..candidates import parse_candidates
from ..io_utils import load_jsonl_by_id, load_problem_statements, load_structure, save_jsonl
from ..selector.base import SelectionContext, score_candidate_against_gold
from ..selector.llm_selector import _card_block
from ..eval.metrics import load_gold

_INSTRUCTION = "Select the single best faulty {level}-level location for the given issue based on the candidates and their evidence."


def _candidate_text(ctx, candidate, card) -> str:
    return _card_block(candidate.index, candidate, card)


def build_dataset(cfg: ECRConfig, level: str, out_dir: str, with_evidence: bool = True,
                  limit: Optional[int] = None) -> dict:
    pool = load_jsonl_by_id(cfg.pool_path(level))
    iids = list(pool)[:limit] if limit else list(pool)
    gold = load_gold(level, cfg.gold_path(level))
    try:
        ps = load_problem_statements(cfg.dataset, cfg.split)
    except Exception:
        ps = {}
    builder = None
    if with_evidence:
        from ..evidence.evidence_card import EvidenceBuilder
        builder = EvidenceBuilder.from_config(cfg)

    sft, pairwise = [], []
    n_pos = n_neg = 0
    for iid in iids:
        try:
            if iid not in gold:
                continue
            structure = load_structure(iid, cfg.project_file_loc)
            cands = parse_candidates(pool[iid], level, structure, sample=cfg.sample)
            if not cands:
                continue
            ctx = SelectionContext(instance_id=iid, level=level, problem_statement=ps.get(iid, ""),
                                   structure=structure)
            cards = builder.build_cards(cands, ctx) if builder else [None] * len(cands)
            labeled = []
            for c, card in zip(cands, cards):
                recall = score_candidate_against_gold(level, c.result_fields(), gold[iid])[0]
                labeled.append((c, card, recall))
            positives = [x for x in labeled if x[2] > 0]
            negatives = [x for x in labeled if x[2] == 0]
            n_pos += len(positives); n_neg += len(negatives)
            if not positives:
                continue
            issue = ps.get(iid, "")
            listing = "\n\n".join(_candidate_text(ctx, c, card) for c, card, _ in labeled)
            best = max(labeled, key=lambda x: x[2])
            sft.append({
                "instance_id": iid,
                "instruction": _INSTRUCTION.format(level=level),
                "input": f"### Issue\n{issue[:6000]}\n\n### Candidates\n{listing}",
                "output": {"selected_index": best[0].index},
            })
            # pairwise: each positive vs a few negatives
            for pc, pcard, _ in positives:
                for nc, ncard, _ in negatives[:3]:
                    pairwise.append({
                        "instance_id": iid,
                        "chosen": f"### Issue\n{issue[:4000]}\n\n{_candidate_text(ctx, pc, pcard)}",
                        "rejected": f"### Issue\n{issue[:4000]}\n\n{_candidate_text(ctx, nc, ncard)}",
                    })
        finally:
            load_structure.cache_clear()
    os.makedirs(out_dir, exist_ok=True)
    sft_path = os.path.join(out_dir, f"sft_{level}.jsonl")
    pw_path = os.path.join(out_dir, f"pairwise_{level}.jsonl")
    save_jsonl(sft_path, sft)
    save_jsonl(pw_path, pairwise)
    stats = {"instances": len(iids), "sft": len(sft), "pairwise": len(pairwise),
             "positives": n_pos, "negatives": n_neg, "sft_path": sft_path, "pairwise_path": pw_path}
    print(f"[build] level={level} -> SFT={len(sft)} pairwise={len(pairwise)} "
          f"(pos={n_pos}, neg={n_neg}); wrote {sft_path}, {pw_path}")
    return stats


def train(pairwise_path: str, base_model: str, out_dir: str, epochs: int = 1, lr: float = 1e-5):
    """Bradley-Terry pairwise training of a scalar reward head. GPU-gated (torch/transformers)."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        print("[train] torch/transformers not available — install them to train. "
              f"({exc})\nThen: ./run_ecr.sh py -m ecr_hifl.selector.train_reward_model train "
              "--pairwise <path> --base_model <hf-id> --out_dir <dir>")
        return

    from .rule_selector import RuleSelector  # noqa: F401  (ensure package import path is valid)
    import json

    rows = [json.loads(l) for l in open(pairwise_path)]
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()

    class PW(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i): return rows[i]

    def collate(batch):
        chosen = tok([b["chosen"] for b in batch], return_tensors="pt", padding=True,
                     truncation=True, max_length=2048)
        rejected = tok([b["rejected"] for b in batch], return_tensors="pt", padding=True,
                       truncation=True, max_length=2048)
        return chosen, rejected

    dl = DataLoader(PW(), batch_size=2, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        for chosen, rejected in dl:
            chosen = {k: v.to(device) for k, v in chosen.items()}
            rejected = {k: v.to(device) for k, v in rejected.items()}
            sc = model(**chosen).logits.squeeze(-1)
            sr = model(**rejected).logits.squeeze(-1)
            loss = -torch.nn.functional.logsigmoid(sc - sr).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[train] epoch {ep} loss={loss.item():.4f}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    print(f"[train] saved reward model -> {out_dir} (set rm_model_path to it, selector=rm)")


def main(argv=None):
    p = argparse.ArgumentParser(description="ECR reward-model dataset / training")
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="construct SFT + pairwise datasets (offline)")
    b.add_argument("--config"); b.add_argument("--level", default="file")
    b.add_argument("--out_dir", default="ecr_hifl/results/rm_data")
    b.add_argument("--no_evidence", action="store_true"); b.add_argument("--limit", type=int)
    t = sub.add_parser("train", help="train a pairwise reward model (GPU)")
    t.add_argument("--pairwise", required=True); t.add_argument("--base_model", required=True)
    t.add_argument("--out_dir", default="ecr_hifl/results/rm_ckpt")
    t.add_argument("--epochs", type=int, default=1); t.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args(argv)
    if args.mode == "build":
        cfg = ECRConfig.load(args.config)
        build_dataset(cfg, args.level, args.out_dir, with_evidence=not args.no_evidence, limit=args.limit)
    else:
        train(args.pairwise, args.base_model, args.out_dir, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
