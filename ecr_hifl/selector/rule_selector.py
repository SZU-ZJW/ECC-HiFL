"""v1 rule selector: weighted sum of evidence signals.

``score = w_rm·rm + w_semantic·semantic + w_graph·graph + w_summary·summary
          + w_history·history + w_verification·verification``

- ``rm``: real generative-RM score if available, else the self-consistency vote (offline surrogate).
- ``semantic``: TF-IDF issue↔candidate similarity.
- ``graph`` / ``summary``: normalized import-graph centrality / issue↔summary overlap.
- ``history`` / ``verification``: git-bugfix signal / LLM plausibility (no-ops offline).

Weights are renormalized over the *active* signals so ablations (turning evidence off, or a
builder that is unavailable offline) don't silently inject constant noise. ``restrict_to`` gates
which evidence groups count, enabling a fast ablation that scores prebuilt cards many ways.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .base import Selection, SelectionContext, Selector

# weight-key -> the evidence group that must be present/active for it to count
_KEY_GROUP = {"rm": "semantic", "semantic": "semantic", "graph": "graph",
              "summary": "summary", "history": "history", "verification": "verification"}


def _features(card) -> Dict[str, float]:
    sem = card.evidence.get("semantic", {})
    rm = card.rm_score if "rm_score" in sem else sem.get("vote_score", 0.0)
    return {
        "rm": float(rm),
        "semantic": float(sem.get("tfidf_score", 0.0)),
        "graph": float(card.evidence.get("graph", {}).get("score", 0.0)),
        "summary": float(card.evidence.get("summary", {}).get("score", 0.0)),
        "history": float(card.evidence.get("history", {}).get("score", 0.0)),
        "verification": float(card.evidence.get("verification", {}).get("score", 0.5)),
    }


def _active_keys(cards: List, restrict: Optional[Set[str]] = None) -> List[str]:
    keys = ["rm", "semantic"]  # semantic group is always built
    def allowed(group):
        return restrict is None or group in restrict
    if allowed("graph") and any("graph" in c.evidence for c in cards):
        keys.append("graph")
    if allowed("summary") and any("summary" in c.evidence for c in cards):
        keys.append("summary")
    if allowed("history") and any(c.evidence.get("history", {}).get("available") for c in cards):
        keys.append("history")
    if allowed("verification") and any(c.evidence.get("verification", {}).get("available") for c in cards):
        keys.append("verification")
    return keys


class RuleSelector(Selector):
    name = "rule"

    def __init__(self, weights: Dict[str, float], restrict_to: Optional[Set[str]] = None):
        self.weights = dict(weights)
        self.restrict_to = set(restrict_to) if restrict_to is not None else None

    def predict(self, candidates: List, ctx: SelectionContext) -> Selection:
        cards = ctx.cards
        if not candidates or not cards:
            return Selection(index=None, fields=self.empty_fields(ctx.level))
        active = _active_keys(cards, self.restrict_to)
        total_w = sum(self.weights.get(k, 0.0) for k in active) or 1.0
        norm = {k: self.weights.get(k, 0.0) / total_w for k in active}
        scores = []
        for card in cards:
            feats = _features(card)
            scores.append(sum(norm[k] * feats[k] for k in active))
        best = max(range(len(scores)), key=lambda i: scores[i])
        return Selection(index=best, fields=candidates[best].result_fields(),
                         scores=[round(s, 5) for s in scores],
                         meta={"active": active, "norm_weights": {k: round(v, 3) for k, v in norm.items()}})
