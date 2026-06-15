"""Experiment drivers: baseline tables and the evidence×N ablation grid.

Both reuse a single per-instance "prep" (parse candidates + build all evidence cards once) and
then score it many ways, so the ablation grid (levels × sample_nums × evidence_sets) doesn't
rebuild graphs for every cell. Offline, ``history`` / ``verification`` are no-ops and their rows
are reported as inactive (honest about what the offline run actually exercises).
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .config import ECRConfig
from .candidates import parse_candidates
from .io_utils import load_jsonl_by_id, load_problem_statements, load_structure
from .selector.base import SelectionContext
from .selector.baselines import make_baseline
from .selector.rule_selector import RuleSelector
from .eval.metrics import compute_metrics, load_gold


# --------------------------------------------------------------- per-instance prep

def _all_evidence_builder(cfg: ECRConfig):
    from .evidence.evidence_card import EvidenceBuilder

    c = copy.deepcopy(cfg)
    c.evidence = {k: True for k in ("semantic", "graph", "summary", "history", "verification")}
    return EvidenceBuilder.from_config(c)


def _prep(cfg, level, sample, iids, pool, ps, build_cards, num_threads):
    """Return {iid: (candidates, ctx, cards)} with cards built once (all evidence)."""
    builder = _all_evidence_builder(cfg) if build_cards else None

    def one(iid):
        structure = load_structure(iid, cfg.project_file_loc)
        cands = parse_candidates(pool[iid], level, structure, sample=sample)
        ctx = SelectionContext(instance_id=iid, level=level, problem_statement=ps.get(iid, ""),
                               structure=structure,
                               extras={"repo": structure.repo if structure else "",
                                       "base_commit": structure.base_commit if structure else ""})
        cards = builder.build_cards(cands, ctx) if (builder and cands) else []
        return iid, (cands, ctx, cards)

    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            return dict(ex.map(one, iids))
    return dict(one(iid) for iid in iids)


def _predict_all(prep, selector, gold=None) -> Dict[str, dict]:
    preds = {}
    for iid, (cands, ctx, cards) in prep.items():
        ctx.cards = cards
        ctx.gold = gold.get(iid) if gold else None
        preds[iid] = selector.predict(cands, ctx).fields
    return preds


# --------------------------------------------------------------- formatting

def _fmt(name: str, m: dict) -> str:
    if m["level"] == "file":
        return (f"  {name:24s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                f"EM={m['em']:.3f} T1={m['top1']:.3f} T3={m['top3']:.3f} T5={m['top5']:.3f} (n={m['n']})")
    return (f"  {name:24s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} | "
            f"microF1={m['micro_f1']:.3f} (n={m['n']})")


# --------------------------------------------------------------- runners

def baseline_table(cfg: ECRConfig, level: str, limit=None, num_threads=1) -> Dict[str, dict]:
    pool = load_jsonl_by_id(cfg.pool_path(level))
    iids = list(pool)[:limit] if limit else list(pool)
    gold = load_gold(level, cfg.gold_path(level))
    try:
        ps = load_problem_statements(cfg.dataset, cfg.split)
    except Exception:
        ps = {}
    prep = _prep(cfg, level, cfg.sample, iids, pool, ps, build_cards=True, num_threads=num_threads)
    print(f"\n=== {level.upper()} baseline table (dataset={cfg.dataset}, sample={cfg.sample}, n={len(iids)}) ===")
    out = {}
    for bname in ("first", "random", "majority", "oracle"):
        m = compute_metrics(level, _predict_all(prep, make_baseline(bname),
                                                gold if bname == "oracle" else None), gold)
        out[bname] = m
        print(_fmt(bname, m))
    m = compute_metrics(level, _predict_all(prep, RuleSelector(cfg.weights)), gold)
    out["rule-ECR(all)"] = m
    print(_fmt("rule-ECR(all)", m))
    return out


def run_ablation(cfg: ECRConfig, limit=None, num_threads=1):
    spec = getattr(cfg, "ablation", {}) or {}
    levels = spec.get("levels", ["file"])
    sample_nums = spec.get("sample_nums", [None])
    baselines = spec.get("baselines", ["first", "majority", "oracle"])
    evidence_sets = spec.get("evidence_sets", [{"name": "all", "evidence": ["graph", "summary"]}])

    for level in levels:
        pool = load_jsonl_by_id(cfg.pool_path(level))
        iids = list(pool)[:limit] if limit else list(pool)
        gold = load_gold(level, cfg.gold_path(level))
        try:
            ps = load_problem_statements(cfg.dataset, cfg.split)
        except Exception:
            ps = {}
        for N in sample_nums:
            prep = _prep(cfg, level, N, iids, pool, ps, build_cards=True, num_threads=num_threads)
            print(f"\n=== ABLATION level={level} sample={N} (n={len(iids)}) ===")
            for bname in baselines:
                m = compute_metrics(level, _predict_all(prep, make_baseline(bname),
                                                        gold if bname == "oracle" else None), gold)
                print(_fmt(bname, m))
            for es in evidence_sets:
                sel = RuleSelector(cfg.weights, restrict_to=set(es["evidence"]) | {"semantic"})
                m = compute_metrics(level, _predict_all(prep, sel), gold)
                print(_fmt(f"rule[{es['name']}]", m))


def main(argv=None):
    p = argparse.ArgumentParser(description="ECR-HiFL experiments (baseline table / ablation grid)")
    p.add_argument("mode", choices=["baseline", "ablation"])
    p.add_argument("--config", help="configs/ecr_*.yaml")
    p.add_argument("--level", choices=["file", "function", "line"])
    p.add_argument("--sample", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--repo_source", help="dir of shared SWE-bench clones -> enables ephemeral history checkout")
    p.add_argument("--num_threads", type=int, default=1)
    args = p.parse_args(argv)
    cfg = ECRConfig.load(args.config)
    if args.sample is not None:
        cfg.sample = args.sample
    if args.repo_source:
        cfg.repo_source = args.repo_source
    if args.mode == "baseline":
        levels = [args.level] if args.level else ["file", "function", "line"]
        for lvl in levels:
            baseline_table(cfg, lvl, limit=args.limit, num_threads=args.num_threads)
    else:
        run_ablation(cfg, limit=args.limit, num_threads=args.num_threads)


if __name__ == "__main__":
    main()
