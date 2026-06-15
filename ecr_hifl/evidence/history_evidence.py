"""History evidence: software-engineering signals from git history.

Features per candidate: ``bugfix_commit_count`` (offline-feasible), plus
``recent_modified_days`` / ``co_changed_files`` as future extensions. Each candidate's score is
the min-max-normalized total bugfix-commit count over its files.

Computing this needs a repo at ``base_commit``. Two sources are supported, in priority order:

1. an **already-checked-out** working tree — ``ctx.extras['repo_path']`` or a per-instance tree
   under ``cfg.repo_root`` (no cloning);
2. an **ephemeral checkout** from a dir of shared clones (``cfg.repo_source``): copy ->
   ``checkout base_commit`` -> read -> delete, memoized per instance (see ``repo_checkout``).

If neither is configured, this builder **degrades gracefully to zeros** (a no-op that doesn't
move scores). It is the evidence whose source is most independent of graph/summary, so it is the
cleanest ablation lever for "multi-source evidence is complementary".
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..candidates import Candidate
from ..io_utils import iter_jsonl
from ..repo_checkout import bugfix_count, instance_bugfix_counts
from .evidence_card import EvidenceBuilderBase

_MAX_FILES_PER_CAND = 5
_COUNTS_CACHE: Dict[str, Dict[str, Dict[str, int]]] = {}


def _resolve_checked_out_path(ctx, cfg) -> Optional[str]:
    """An already-checked-out working tree for this instance (no clone), or None."""
    p = (ctx.extras or {}).get("repo_path")
    if p and os.path.isdir(os.path.join(p, ".git")):
        return p
    root = getattr(cfg, "repo_root", None) if cfg else None
    if not root:
        return None
    repo = (ctx.extras or {}).get("repo", "")
    for cand in (ctx.instance_id, repo.replace("/", "__"), repo.split("/")[-1] if repo else ""):
        cp = os.path.join(root, cand) if cand else None
        if cp and os.path.isdir(os.path.join(cp, ".git")):
            return cp
    return None


class HistoryEvidenceBuilder(EvidenceBuilderBase):
    name = "history"

    def __init__(self, cfg=None):
        self.cfg = cfg

    def _precomputed_counts(self, files: List[str], ctx) -> Optional[Dict[str, int]]:
        path = getattr(self.cfg, "history_counts_path", None) if self.cfg else None
        if not path:
            return None
        cache = _COUNTS_CACHE.get(path)
        if cache is None:
            cache = {}
            try:
                for row in iter_jsonl(path):
                    if row.get("available", True):
                        cache[row["instance_id"]] = {
                            str(k): int(v) for k, v in (row.get("counts") or {}).items()
                        }
            except FileNotFoundError:
                cache = {}
            _COUNTS_CACHE[path] = cache
        counts = cache.get(ctx.instance_id)
        if counts is None:
            return None
        return {f: counts.get(f, 0) for f in files}

    def _counts_by_file(self, candidates: List[Candidate], ctx) -> Optional[Dict[str, int]]:
        """Return ``{file: bugfix_count}`` for every candidate file, or None if no repo source."""
        files = list(dict.fromkeys(
            f for c in candidates for f in list(dict.fromkeys(c.files))[:_MAX_FILES_PER_CAND]
        ))
        if not files:
            return None
        # 0) precomputed history counts -> fastest path for repeated ablations/cards
        precomputed = self._precomputed_counts(files, ctx)
        if precomputed is not None:
            return precomputed
        # 1) already-checked-out tree -> read directly, no clone
        repo_path = _resolve_checked_out_path(ctx, self.cfg)
        if repo_path is not None:
            return {f: bugfix_count(repo_path, f) for f in files}
        # 2) ephemeral copy -> checkout(base_commit) -> delete from a shared clone source
        repo_source = getattr(self.cfg, "repo_source", None) if self.cfg else None
        if repo_source:
            extras = ctx.extras or {}
            counts = instance_bugfix_counts(
                repo_source, extras.get("repo", ""), ctx.instance_id,
                extras.get("base_commit", ""), files,
                tmp_parent=getattr(self.cfg, "checkout_tmp", None),
            )
            return counts or None  # {} (unavailable) -> no-op
        # 3) nothing configured
        return None

    def build_all(self, candidates: List[Candidate], ctx) -> List[dict]:
        if not candidates:
            return []
        counts = self._counts_by_file(candidates, ctx)
        if counts is None:
            # offline / no checkout: neutral no-op so this evidence simply doesn't move scores
            return [{"score": 0.0, "available": False} for _ in candidates]
        raw, rich = [], []
        for c in candidates:
            files = list(dict.fromkeys(c.files))[:_MAX_FILES_PER_CAND]
            per = {f: counts.get(f, 0) for f in files}
            total = sum(per.values())
            raw.append(float(total))
            rich.append({"available": True, "bugfix_commit_count": total, "per_file_bugfix": per})
        lo, hi = (min(raw), max(raw)) if raw else (0.0, 0.0)
        span = (hi - lo) or 1.0
        for r, v in zip(rich, raw):
            r["score"] = round((v - lo) / span, 4)
        return rich
