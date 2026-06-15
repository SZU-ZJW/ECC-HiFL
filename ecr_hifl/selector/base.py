"""Selector interface shared by baselines and evidence-calibrated selectors.

A selector turns a list of :class:`~ecr_hifl.candidates.Candidate` (plus optional evidence
cards in the :class:`SelectionContext`) into a :class:`Selection`: either a chosen candidate
index, or a fused ``fields`` dict (e.g. majority voting). The pipeline merges ``Selection.fields``
into the per-instance output row (upstream ``loc_outputs.jsonl`` shape).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SelectionContext:
    instance_id: str
    level: str
    problem_statement: str = ""
    structure: Any = None                 # ecr_hifl.io_utils.RepoStructure | None
    cards: Optional[List[Any]] = None     # ecr_hifl.evidence.evidence_card.EvidenceCard, aligned w/ candidates
    gold: Any = None                      # gold for this instance (oracle only)
    extras: dict = field(default_factory=dict)


@dataclass
class Selection:
    index: Optional[int]                  # chosen 0-based candidate index, or None for a fused prediction
    fields: dict                          # result fields to merge into the output row
    scores: Optional[List[float]] = None  # per-candidate scores, if any
    meta: dict = field(default_factory=dict)


class Selector(ABC):
    name: str = "selector"

    @abstractmethod
    def predict(self, candidates: List, ctx: SelectionContext) -> Selection:
        ...

    @staticmethod
    def empty_fields(level: str) -> dict:
        return {"file": {"found_files": []},
                "function": {"found_files": [], "found_related_locs": {}},
                "line": {"found_files": [], "found_edit_locs": {}}}[level]


def score_candidate_against_gold(level: str, fields: dict, gold_one) -> tuple:
    """(recall, f1, precision) of a single candidate's ``fields`` vs one instance's gold.

    Used by the oracle baseline. Reuses the real metric functions on a one-element map.
    """
    from ..eval.metrics import compute_metrics

    if gold_one is None:
        return (0.0, 0.0, 0.0)
    preds = {"_": dict(fields)}
    gold = {"_": gold_one}
    m = compute_metrics(level, preds, gold)
    return (m["recall"], m["f1"], m["precision"])
