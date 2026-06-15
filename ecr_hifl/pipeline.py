"""End-to-end ECR-HiFL pipeline: pool -> candidates -> evidence cards -> selector -> results.

Produces a results jsonl in the upstream ``loc_outputs.jsonl`` shape (so it is directly
eval-able by ``ecr_hifl.eval`` and comparable to existing HiLoRM runs), with an extra ``ecr``
field recording the selector, per-candidate scores, and chosen index.

Run via ``./run_ecr.sh select|baseline ...`` (which sets PYTHONPATH / PROJECT_FILE_LOC / HILORM_*).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .candidates import parse_candidates
from .config import ECRConfig
from .evidence.evidence_card import EvidenceCard
from .io_utils import load_jsonl_by_id, load_problem_statements, load_structure, save_jsonl
from .selector.base import Selection, SelectionContext, Selector


# --------------------------------------------------------------------- selector factory

def make_selector(cfg: ECRConfig, name: Optional[str] = None) -> Selector:
    name = name or cfg.selector_type
    from .selector.baselines import BASELINES, make_baseline

    if name in BASELINES:
        return make_baseline(name)
    if name == "rule":
        from .selector.rule_selector import RuleSelector
        return RuleSelector(cfg.weights)
    if name == "llm":
        from .selector.llm_selector import LLMSelector
        return LLMSelector(cfg)
    if name == "rm":
        from .selector.rm_selector import RMSelector
        return RMSelector(cfg)
    raise ValueError(f"unknown selector {name!r}")


def _needs_cards(name: str) -> bool:
    return name in ("rule", "llm", "rm")


def _cached_cards_for(row: Optional[dict], level: str, sample: Optional[int], candidates: List) -> Optional[List[EvidenceCard]]:
    """Return cached evidence cards if they align with the current candidate list."""
    if not row or row.get("level") != level or row.get("sample") != sample:
        return None
    raw_cards = row.get("cards") or []
    if len(raw_cards) != len(candidates):
        return None
    cards: List[EvidenceCard] = []
    for raw, cand in zip(raw_cards, candidates):
        if raw.get("candidate_id") != cand.index:
            return None
        if list(raw.get("files") or []) != list(cand.files):
            return None
        cards.append(EvidenceCard(
            candidate_index=raw.get("candidate_id", cand.index),
            level=raw.get("level", level),
            files=list(raw.get("files") or []),
            symbols=list(raw.get("symbols") or []),
            rm_score=float(raw.get("rm_score", 0.0) or 0.0),
            evidence=dict(raw.get("evidence") or {}),
        ))
    return cards


# --------------------------------------------------------------------- core run

def run_pipeline(cfg: ECRConfig, selector_name: Optional[str] = None, *, limit: Optional[int] = None,
                 num_threads: int = 1, verbose: bool = True,
                 cards_cache: Optional[Dict[str, dict]] = None) -> List[dict]:
    level = cfg.level
    selector_name = selector_name or cfg.selector_type
    selector = make_selector(cfg, selector_name)
    needs_cards = _needs_cards(selector_name)
    needs_gold = selector_name == "oracle"

    pool = load_jsonl_by_id(cfg.pool_path(level))
    iids = list(pool)
    if limit:
        iids = iids[:limit]

    problem_statements = {}
    try:
        problem_statements = load_problem_statements(cfg.dataset, cfg.split)
    except Exception as exc:  # offline w/o HF cache: degrade to empty issue text
        if verbose:
            print(f"[warn] could not load problem statements ({exc}); semantic/summary will be neutral")

    gold = {}
    if needs_gold:
        from .eval.metrics import load_gold
        gold = load_gold(level, cfg.gold_path(level))

    builder = None
    if needs_cards:
        from .evidence.evidence_card import EvidenceBuilder
        builder = EvidenceBuilder.from_config(cfg)

    def process(iid: str) -> dict:
        try:
            row = pool[iid]
            structure = load_structure(iid, cfg.project_file_loc)
            candidates = parse_candidates(row, level, structure, sample=cfg.sample)
            ctx = SelectionContext(
                instance_id=iid, level=level,
                problem_statement=problem_statements.get(iid, ""),
                structure=structure,
                gold=gold.get(iid) if needs_gold else None,
                extras={"repo": (structure.repo if structure else ""),
                        "base_commit": (structure.base_commit if structure else "")},
            )
            if needs_cards and candidates:
                cached_cards = _cached_cards_for(cards_cache.get(iid) if cards_cache else None,
                                                 level, cfg.sample, candidates)
                ctx.cards = cached_cards if cached_cards is not None else builder.build_cards(candidates, ctx)
            sel: Selection = selector.predict(candidates, ctx)
            out = {"instance_id": iid, **sel.fields,
                   "ecr": {"selector": selector_name, "level": level,
                           "chosen_index": sel.index, "n_candidates": len(candidates),
                           "scores": sel.scores, "meta": sel.meta}}
            return out
        finally:
            load_structure.cache_clear()

    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            results = list(ex.map(process, iids))
    else:
        results = [process(iid) for iid in iids]
    return results


def write_results(cfg: ECRConfig, selector_name: str, results: List[dict]) -> str:
    out_dir = os.path.join(cfg.output_dir, f"{cfg.level}_{selector_name}")
    path = os.path.join(out_dir, "loc_outputs.jsonl")
    save_jsonl(path, results)
    return path


# --------------------------------------------------------------------- CLI

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ECR-HiFL selection pipeline")
    p.add_argument("--config", help="path to configs/ecr_*.yaml")
    p.add_argument("--level", choices=["file", "function", "line"])
    p.add_argument("--selector", help="first|random|majority|oracle|rule|llm|rm")
    p.add_argument("--evidence", dest="evidence_list", help="comma list: graph,summary,history,verification")
    p.add_argument("--sample", type=int, help="N in Best-of-N (truncate pool)")
    p.add_argument("--dataset")
    p.add_argument("--pool")
    p.add_argument("--gold")
    p.add_argument("--project_file_loc")
    p.add_argument("--repo_source", help="dir of shared SWE-bench clones -> enables ephemeral history checkout")
    p.add_argument("--output_dir")
    p.add_argument("--cards_cache", help="precomputed evidence-card JSONL for this level/sample")
    p.add_argument("--limit", type=int, help="only process first N instances (debug)")
    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--eval", action="store_true", help="evaluate the produced results against gold")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    cfg = ECRConfig.load(args.config)
    cfg.apply_overrides(
        level=args.level, dataset=args.dataset, sample=args.sample,
        project_file_loc=args.project_file_loc, output_dir=args.output_dir,
        evidence_list=args.evidence_list, selector_type=args.selector,
    )
    if args.pool:
        cfg.pools[cfg.level] = args.pool
    if args.gold:
        cfg.gold[cfg.level] = args.gold
    if args.repo_source:
        cfg.repo_source = args.repo_source
    selector_name = args.selector or cfg.selector_type
    cards_cache = load_jsonl_by_id(args.cards_cache) if args.cards_cache else None

    print(f"== ECR-HiFL: level={cfg.level} selector={selector_name} "
          f"evidence={cfg.enabled_evidence()} sample={cfg.sample} ==")
    results = run_pipeline(cfg, selector_name, limit=args.limit, num_threads=args.num_threads,
                           cards_cache=cards_cache)
    path = write_results(cfg, selector_name, results)
    print(f"wrote {len(results)} predictions -> {path}")

    if args.eval:
        from .eval.metrics import compute_metrics, format_metrics, load_gold
        gold = load_gold(cfg.level, cfg.gold_path())
        preds = {r["instance_id"]: r for r in results}
        print(format_metrics(compute_metrics(cfg.level, preds, gold)))


if __name__ == "__main__":
    main()
