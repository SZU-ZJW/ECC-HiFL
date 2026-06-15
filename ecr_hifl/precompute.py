"""Precompute reusable ECR-HiFL artifacts.

This module prepares expensive or frequently inspected intermediate data before running large
experiment grids:

- repo graph indexes and structural file summaries;
- per-instance candidate evidence cards for selected levels and sample sizes.

The outputs are JSONL files under ``ecr_hifl/cache/<dataset>/`` by default. They are intentionally
plain artifacts first: experiments can be audited from them, and future pipeline changes can load
them as caches without changing the raw candidate pools.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional

from .candidates import LEVELS, parse_candidates
from .config import ECRConfig
from .evidence.evidence_card import EvidenceBuilder
from .evidence.graph_evidence import build_import_graph
from .evidence.summary_evidence import structural_summary
from .io_utils import load_jsonl_by_id, load_problem_statements, load_structure, save_jsonl
from .repo_checkout import instance_bugfix_counts
from .selector.base import SelectionContext


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()


def _parse_csv(value: Optional[str], default: Iterable[str]) -> List[str]:
    if not value:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_samples(value: Optional[str], default: Optional[int]) -> List[Optional[int]]:
    if value is None:
        return [default]
    out: List[Optional[int]] = []
    for part in value.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p in ("all", "none", "null"):
            out.append(None)
        else:
            out.append(int(p))
    return out or [default]


def _sample_name(sample: Optional[int]) -> str:
    return "all" if sample is None else str(sample)


def _base_out_dir(cfg: ECRConfig, out_dir: Optional[str]) -> str:
    root = out_dir or os.path.join("ecr_hifl", "cache")
    return os.path.join(root, _slug(cfg.dataset), cfg.split)


def _completed_ids(path: str, level: Optional[str] = None, sample: Optional[int] = None) -> set:
    """Return completed instance ids from an existing JSONL and trim a partial tail line."""
    if not os.path.isfile(path):
        return set()
    done = set()
    last_good = 0
    with open(path, "rb+") as f:
        while True:
            line = f.readline()
            if not line:
                last_good = f.tell()
                break
            if not line.strip():
                last_good = f.tell()
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except Exception:
                break
            if (level is None or row.get("level") == level) and row.get("sample") == sample:
                iid = row.get("instance_id")
                if iid:
                    done.add(iid)
            last_good = f.tell()
        f.truncate(last_good)
    return done


def _append_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def _instance_ids(cfg: ECRConfig, levels: List[str], limit: Optional[int]) -> List[str]:
    seen, out = set(), []
    for level in levels:
        pool = load_jsonl_by_id(cfg.pool_path(level))
        for iid in pool:
            if iid in seen:
                continue
            seen.add(iid)
            out.append(iid)
            if limit and len(out) >= limit:
                return out
    return out


def _graph_row_uncached(cfg: ECRConfig, iid: str) -> dict:
    structure = load_structure(iid, cfg.project_file_loc)
    if structure is None:
        return {"instance_id": iid, "error": "missing structure"}
    try:
        graph = build_import_graph(structure)
        degree = {node: graph.in_degree(node) + graph.out_degree(node) for node in graph.nodes}
        py_files = [f for f in structure.file_paths if f.endswith(".py")]
        summaries = {f: structural_summary(structure, [f]) for f in py_files}
        top_degree = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
        return {
            "instance_id": iid,
            "repo": structure.repo,
            "base_commit": structure.base_commit,
            "n_files": len(structure.file_paths),
            "n_python_files": len(py_files),
            "n_import_edges": graph.number_of_edges(),
            "import_edges": [[src, dst] for src, dst in graph.edges],
            "degree": degree,
            "top_degree_files": [{"file": f, "degree": d} for f, d in top_degree],
            "symbols_by_file": structure.symbols_by_file,
            "structural_summary_by_file": summaries,
        }
    except Exception as exc:
        return {"instance_id": iid, "repo": structure.repo, "error": f"{type(exc).__name__}: {exc}"}


def _graph_row(cfg: ECRConfig, iid: str) -> dict:
    try:
        return _graph_row_uncached(cfg, iid)
    finally:
        load_structure.cache_clear()


def precompute_graphs(cfg: ECRConfig, levels: List[str], out_dir: Optional[str] = None,
                      limit: Optional[int] = None, num_threads: int = 1) -> str:
    iids = _instance_ids(cfg, levels, limit)
    path = os.path.join(_base_out_dir(cfg, out_dir), "repo_graphs.jsonl")
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            save_jsonl(path, ex.map(lambda iid: _graph_row(cfg, iid), iids))
    else:
        save_jsonl(path, (_graph_row(cfg, iid) for iid in iids))
    print(f"[precompute] graphs: wrote {len(iids)} rows -> {path}")
    return path


def _candidate_dict(candidate) -> dict:
    return {
        "index": candidate.index,
        "level": candidate.level,
        "files": candidate.files,
        "elements": candidate.elements,
        "raw_preview": candidate.raw_text[:500],
    }


def _cards_for_instance_uncached(cfg: ECRConfig, level: str, sample: Optional[int], pool: Dict[str, dict],
                                 problem_statements: Dict[str, str], builder: EvidenceBuilder, iid: str) -> dict:
    structure = load_structure(iid, cfg.project_file_loc)
    candidates = parse_candidates(pool[iid], level, structure, sample=sample)
    ctx = SelectionContext(
        instance_id=iid,
        level=level,
        problem_statement=problem_statements.get(iid, ""),
        structure=structure,
        extras={
            "repo": structure.repo if structure else "",
            "base_commit": structure.base_commit if structure else "",
        },
    )
    cards = builder.build_cards(candidates, ctx) if candidates else []
    return {
        "instance_id": iid,
        "level": level,
        "sample": sample,
        "repo": structure.repo if structure else "",
        "base_commit": structure.base_commit if structure else "",
        "n_candidates": len(candidates),
        "candidates": [_candidate_dict(c) for c in candidates],
        "cards": [c.to_dict() for c in cards],
    }


def _cards_for_instance(cfg: ECRConfig, level: str, sample: Optional[int], pool: Dict[str, dict],
                        problem_statements: Dict[str, str], builder: EvidenceBuilder, iid: str) -> dict:
    try:
        return _cards_for_instance_uncached(cfg, level, sample, pool, problem_statements, builder, iid)
    finally:
        load_structure.cache_clear()


def precompute_cards(cfg: ECRConfig, levels: List[str], samples: List[Optional[int]],
                     out_dir: Optional[str] = None, limit: Optional[int] = None,
                     num_threads: int = 1, resume: bool = False) -> List[str]:
    try:
        problem_statements = load_problem_statements(cfg.dataset, cfg.split)
    except Exception as exc:
        print(f"[precompute] warning: problem statements unavailable ({type(exc).__name__}); using empty issue text")
        problem_statements = {}

    builder = EvidenceBuilder.from_config(cfg)
    paths: List[str] = []
    for level in levels:
        pool = load_jsonl_by_id(cfg.pool_path(level))
        iids = list(pool)[:limit] if limit else list(pool)
        for sample in samples:
            path = os.path.join(
                _base_out_dir(cfg, out_dir),
                "evidence_cards",
                f"{level}_sample_{_sample_name(sample)}.jsonl",
            )
            run_iids = iids
            if resume:
                done = _completed_ids(path, level, sample)
                run_iids = [iid for iid in iids if iid not in done]
                if done:
                    print(f"[precompute] resume: level={level} sample={_sample_name(sample)} "
                          f"found {len(done)} completed rows in {path}")
                if not run_iids:
                    print(f"[precompute] cards: level={level} sample={_sample_name(sample)} "
                          f"already complete ({len(done)} rows) -> {path}")
                    paths.append(path)
                    continue
            if num_threads > 1:
                with ThreadPoolExecutor(max_workers=num_threads) as ex:
                    rows = ex.map(
                        lambda iid: _cards_for_instance(cfg, level, sample, pool, problem_statements, builder, iid),
                        run_iids,
                    )
                    _append_jsonl(path, rows) if resume else save_jsonl(path, rows)
            else:
                rows = (
                    _cards_for_instance(cfg, level, sample, pool, problem_statements, builder, iid)
                    for iid in run_iids
                )
                _append_jsonl(path, rows) if resume else save_jsonl(path, rows)
            paths.append(path)
            print(f"[precompute] cards: level={level} sample={_sample_name(sample)} "
                  f"wrote {len(run_iids)} rows -> {path}")
    return paths


def _max_sample(samples: List[Optional[int]]) -> Optional[int]:
    return None if any(s is None for s in samples) else max((s or 0) for s in samples)


def _history_counts_row(cfg: ECRConfig, levels: List[str], samples: List[Optional[int]],
                        pools: Dict[str, Dict[str, dict]], iid: str) -> dict:
    try:
        structure = load_structure(iid, cfg.project_file_loc)
        if structure is None:
            return {"instance_id": iid, "available": False, "error": "missing structure", "counts": {}}
        sample = _max_sample(samples)
        files: List[str] = []
        for level in levels:
            row = pools.get(level, {}).get(iid)
            if not row:
                continue
            candidates = parse_candidates(row, level, structure, sample=sample)
            for c in candidates:
                files.extend(list(dict.fromkeys(c.files))[:5])
        files = list(dict.fromkeys(files))
        if not files:
            return {
                "instance_id": iid,
                "repo": structure.repo,
                "base_commit": structure.base_commit,
                "available": True,
                "n_files": 0,
                "counts": {},
            }
        counts = instance_bugfix_counts(
            getattr(cfg, "repo_source", None), structure.repo, iid, structure.base_commit, files,
            tmp_parent=getattr(cfg, "checkout_tmp", None),
        )
        return {
            "instance_id": iid,
            "repo": structure.repo,
            "base_commit": structure.base_commit,
            "available": bool(counts),
            "n_files": len(files),
            "counts": counts,
        }
    finally:
        load_structure.cache_clear()


def precompute_history_counts(cfg: ECRConfig, levels: List[str], samples: List[Optional[int]],
                              out_dir: Optional[str] = None, limit: Optional[int] = None,
                              num_threads: int = 1) -> str:
    pools = {level: load_jsonl_by_id(cfg.pool_path(level)) for level in levels}
    seen, iids = set(), []
    for level in levels:
        for iid in pools[level]:
            if iid in seen:
                continue
            seen.add(iid)
            iids.append(iid)
            if limit and len(iids) >= limit:
                break
        if limit and len(iids) >= limit:
            break
    path = os.path.join(_base_out_dir(cfg, out_dir), "history_counts.jsonl")
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            save_jsonl(path, ex.map(lambda iid: _history_counts_row(cfg, levels, samples, pools, iid), iids))
    else:
        save_jsonl(path, (_history_counts_row(cfg, levels, samples, pools, iid) for iid in iids))
    print(f"[precompute] history-counts: wrote {len(iids)} rows -> {path}")
    return path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute reusable ECR-HiFL graph/evidence artifacts")
    p.add_argument("mode", choices=["graphs", "cards", "history-counts", "all"])
    p.add_argument("--config", help="configs/ecr_*.yaml")
    p.add_argument("--levels", help="comma list: file,function,line (default: config level)")
    p.add_argument("--samples", help="comma list: 3,7,15,30,all (cards/all only; default: config sample)")
    p.add_argument("--evidence", dest="evidence_list", help="comma list: graph,summary,history,verification")
    p.add_argument("--project_file_loc", help="repo-structure directory override")
    p.add_argument("--repo_source", help="dir of shared SWE-bench clones -> enables history evidence")
    p.add_argument("--history_counts_path", help="precomputed history_counts.jsonl for cards mode")
    p.add_argument("--out_dir", help="output root (default: ecr_hifl/cache)")
    p.add_argument("--resume", action="store_true", help="append missing card rows instead of overwriting")
    p.add_argument("--limit", type=int, help="debug: only process first N instances")
    p.add_argument("--num_threads", type=int, default=1)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    cfg = ECRConfig.load(args.config)
    if args.evidence_list:
        cfg.apply_overrides(evidence_list=args.evidence_list)
    if args.project_file_loc:
        cfg.project_file_loc = args.project_file_loc
    if args.repo_source:
        cfg.repo_source = args.repo_source
    if args.history_counts_path:
        cfg.history_counts_path = args.history_counts_path
    levels = _parse_csv(args.levels, [cfg.level])
    bad = [lvl for lvl in levels if lvl not in LEVELS]
    if bad:
        raise SystemExit(f"unknown level(s): {bad}; choose from {LEVELS}")
    samples = _parse_samples(args.samples, cfg.sample)

    if args.mode in ("graphs", "all"):
        precompute_graphs(cfg, levels, args.out_dir, args.limit, args.num_threads)
    if args.mode == "history-counts":
        precompute_history_counts(cfg, levels, samples, args.out_dir, args.limit, args.num_threads)
    if args.mode in ("cards", "all"):
        precompute_cards(cfg, levels, samples, args.out_dir, args.limit, args.num_threads, args.resume)


if __name__ == "__main__":
    main()
