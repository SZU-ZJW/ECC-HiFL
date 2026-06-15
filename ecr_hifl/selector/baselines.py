"""Reference baselines: first / random / majority(vote) / oracle.

These need no servers and no evidence. They calibrate what the evidence-calibrated selectors
have to beat:
- ``first``    -> candidate[0] (≈ the vanilla Agentless single-sample default).
- ``random``   -> a deterministic pseudo-random candidate (seeded per instance, reproducible).
- ``majority`` -> frequency fusion across the pool (a.k.a. majority voting / self-consistency).
                  Emits a fused prediction, not a single candidate.
- ``oracle``   -> the best-possible candidate vs gold (an upper bound; uses ctx.gold).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import List

from .base import Selection, SelectionContext, Selector, score_candidate_against_gold


class FirstSelector(Selector):
    name = "first"

    def predict(self, candidates: List, ctx: SelectionContext) -> Selection:
        if not candidates:
            return Selection(index=None, fields=self.empty_fields(ctx.level))
        return Selection(index=0, fields=candidates[0].result_fields())


class RandomSelector(Selector):
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def predict(self, candidates: List, ctx: SelectionContext) -> Selection:
        if not candidates:
            return Selection(index=None, fields=self.empty_fields(ctx.level))
        h = hashlib.md5(f"{self.seed}:{ctx.instance_id}".encode()).hexdigest()
        idx = int(h, 16) % len(candidates)
        return Selection(index=idx, fields=candidates[idx].result_fields(),
                         meta={"seed": self.seed})


class MajoritySelector(Selector):
    """Frequency fusion across the pool.

    file:     rank files by how many candidates include them, keep those at/above a vote
              threshold (default: appear in >= 1 candidate), capped to ``top_k``.
    function/line: union the per-(file,loc) entries that appear in >= ``min_votes`` candidates.
    """
    name = "majority"

    def __init__(self, top_k: int = 5, min_votes: int = 2):
        self.top_k = top_k
        self.min_votes = min_votes

    def predict(self, candidates: List, ctx: SelectionContext) -> Selection:
        if not candidates:
            return Selection(index=None, fields=self.empty_fields(ctx.level))
        if ctx.level == "file":
            votes = Counter()
            order = {}
            for c in candidates:
                for rank, f in enumerate(c.files):
                    votes[f] += 1
                    order.setdefault(f, rank)
            ranked = sorted(votes, key=lambda f: (-votes[f], order[f]))
            return Selection(index=None, fields={"found_files": ranked[: self.top_k]},
                             meta={"votes": dict(votes)})
        # function / line: vote per (file, loc-line)
        votes = defaultdict(Counter)
        for c in candidates:
            for f, locs in c.elements.items():
                for loc in set(locs):
                    votes[f][loc] += 1
        elements = {}
        for f, c in votes.items():
            kept = [loc for loc, v in c.items() if v >= self.min_votes] or [loc for loc, _ in c.most_common(3)]
            if kept:
                elements[f] = kept
        key = "found_related_locs" if ctx.level == "function" else "found_edit_locs"
        fields = {"found_files": list(elements.keys()),
                  key: {f: ["\n".join(locs)] for f, locs in elements.items()}}
        return Selection(index=None, fields=fields)


class OracleSelector(Selector):
    """Upper bound: pick the candidate that maximizes (recall, f1, precision) vs gold."""
    name = "oracle"

    def predict(self, candidates: List, ctx: SelectionContext) -> Selection:
        if not candidates:
            return Selection(index=None, fields=self.empty_fields(ctx.level))
        best_idx, best_score = 0, (-1.0, -1.0, -1.0)
        for i, c in enumerate(candidates):
            s = score_candidate_against_gold(ctx.level, c.result_fields(), ctx.gold)
            if s > best_score:
                best_score, best_idx = s, i
        return Selection(index=best_idx, fields=candidates[best_idx].result_fields(),
                         meta={"oracle_score": best_score})


BASELINES = {
    "first": FirstSelector,
    "random": RandomSelector,
    "majority": MajoritySelector,
    "oracle": OracleSelector,
}


def make_baseline(name: str) -> Selector:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    return BASELINES[name]()
