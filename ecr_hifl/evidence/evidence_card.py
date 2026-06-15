"""Evidence cards and the builder orchestrator.

An :class:`EvidenceCard` stores per-candidate ``semantic`` / ``graph`` / ``history`` /
``verification`` evidence, plus an ``rm_score``.
Each evidence builder returns one rich dict per candidate that always carries a normalized
scalar ``"score"`` in ``[0, 1]`` (so selectors can consume it directly), alongside richer fields
for case studies. Builders see *all* candidates of an instance at once so they can normalize
within the instance (vote fractions, min-max degree, TF-IDF over the candidate set).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..candidates import Candidate
from ..selector.base import SelectionContext


@dataclass
class EvidenceCard:
    candidate_index: int          # 1-based, matches Candidate.index
    level: str
    files: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)   # "function:/class:" lines (function/line)
    rm_score: float = 0.0         # real generative-RM score if available, else 0
    evidence: Dict[str, dict] = field(default_factory=dict)

    def feature(self, group: str, key: str = "score", default: float = 0.0) -> float:
        return float(self.evidence.get(group, {}).get(key, default))

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_index,
            "level": self.level,
            "files": self.files,
            "symbols": self.symbols,
            "rm_score": self.rm_score,
            "evidence": self.evidence,
        }


class EvidenceBuilderBase:
    """Base class for an evidence builder. Subclasses implement :meth:`build_all`."""

    name = "base"

    def build_all(self, candidates: List[Candidate], ctx: SelectionContext) -> List[dict]:
        """Return one evidence dict per candidate (same order); each must contain ``"score"``."""
        return [{"score": 0.0} for _ in candidates]


class EvidenceBuilder:
    """Runs the enabled evidence builders and assembles one card per candidate."""

    def __init__(self, builders: Dict[str, EvidenceBuilderBase]):
        self.builders = builders  # {group_name: builder}

    @classmethod
    def from_config(cls, cfg) -> "EvidenceBuilder":
        """Construct from an :class:`~ecr_hifl.config.ECRConfig` (honors evidence toggles)."""
        from .graph_evidence import GraphEvidenceBuilder
        from .history_evidence import HistoryEvidenceBuilder
        from .semantic_evidence import SemanticEvidenceBuilder
        from .summary_evidence import SummaryEvidenceBuilder
        from .verification_evidence import VerificationEvidenceBuilder

        registry = {
            "semantic": SemanticEvidenceBuilder,
            "graph": GraphEvidenceBuilder,
            "summary": SummaryEvidenceBuilder,
            "history": HistoryEvidenceBuilder,
            "verification": VerificationEvidenceBuilder,
        }
        builders = {}
        # semantic is always built (it provides the offline RM surrogate + tfidf signal)
        enabled = set(cfg.enabled_evidence()) | {"semantic"}
        for name in registry:
            if name in enabled:
                builders[name] = registry[name](cfg)
        return cls(builders)

    def build_cards(self, candidates: List[Candidate], ctx: SelectionContext) -> List[EvidenceCard]:
        cards = [
            EvidenceCard(candidate_index=c.index, level=c.level, files=list(c.files),
                         symbols=[loc for locs in c.elements.values() for loc in locs])
            for c in candidates
        ]
        for group, builder in self.builders.items():
            try:
                sub = builder.build_all(candidates, ctx)
            except Exception as exc:  # one bad builder must not sink the run
                sub = [{"score": 0.0, "error": f"{type(exc).__name__}: {exc}"} for _ in candidates]
            for card, ev in zip(cards, sub):
                card.evidence[group] = ev
        # lift real RM score onto the card if the semantic builder produced one
        for card in cards:
            rm = card.evidence.get("semantic", {}).get("rm_score")
            if rm is not None:
                card.rm_score = float(rm)
        return cards
