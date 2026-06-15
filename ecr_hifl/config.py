"""ECR-HiFL configuration.

Loads a YAML (``configs/ecr_*.yaml``) describing data roots, per-level candidate pools and gold,
evidence toggles, selector type/weights, and server toggles. Server endpoints themselves are NOT
duplicated here — they come from the HiLoRM ``EP`` singleton (``config.yaml`` keyed by
``$HILORM_STAGE``). Defaults use relative placeholder paths under ``data/``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import yaml

# Sensible offline defaults (overridden by the YAML / CLI).
_DEFAULTS: Dict[str, Any] = {
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "split": "test",
    "level": "file",
    "sample": None,                 # N in Best-of-N; None = use the whole pool
    "project_file_loc": "data/repo_structures",
    "repo_root": None,              # dir of already-checked-out per-instance trees (optional; no clone)
    "repo_source": None,            # dir of shared SWE-bench clones (e.g. data/repos) for
                                    # ephemeral per-instance checkout (History/repair); None = disabled
    "checkout_tmp": None,           # parent dir for ephemeral checkouts (default: sibling of repo_source)
    "pools": {
        "file": "data/pools/swe-bench-lite/file_level/loc_outputs.jsonl",
        "function": "data/pools/swe-bench-lite/related_elements/loc_outputs.jsonl",
        "line": "data/pools/swe-bench-lite/edit_location_samples/loc_outputs.jsonl",
    },
    "gold": {
        "file": "data/gold/swe-bench-lite/modified_files.jsonl",
        "function": "data/gold/swe-bench-lite/element_level.jsonl",
        "line": "data/gold/swe-bench-lite/element_level_line_number.jsonl",
    },
    "evidence": {"semantic": True, "graph": True, "summary": True, "history": True, "verification": False},
    "selector": {
        "type": "rule",
        "weights": {"rm": 0.30, "semantic": 0.20, "graph": 0.15, "summary": 0.15,
                    "history": 0.10, "verification": 0.10},
    },
    # server toggles (all off offline; flip on when the corresponding endpoint is reachable)
    "use_rm_server": False,
    "use_summary_server": False,
    "use_verification_server": False,
    "use_llm_judge": False,
    "summary_model": "q7bc",
    "verification_model": "q7bc",
    "judge_model": None,            # None -> use EP.judge_model
    "output_dir": "ecr_hifl/results",
}


class ECRConfig:
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        merged = copy.deepcopy(_DEFAULTS)
        if data:
            _deep_update(merged, data)
        self._data = merged
        for k, v in merged.items():
            setattr(self, k, v)

    @classmethod
    def from_yaml(cls, path: str) -> "ECRConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    @classmethod
    def load(cls, path: Optional[str]) -> "ECRConfig":
        return cls.from_yaml(path) if path else cls()

    # ---- convenience accessors ----
    def enabled_evidence(self) -> List[str]:
        return [k for k, v in (self.evidence or {}).items() if v]

    @property
    def weights(self) -> Dict[str, float]:
        return dict((self.selector or {}).get("weights", {}))

    @property
    def selector_type(self) -> str:
        return (self.selector or {}).get("type", "rule")

    def pool_path(self, level: Optional[str] = None) -> str:
        return self.pools[level or self.level]

    def gold_path(self, level: Optional[str] = None) -> str:
        return self.gold[level or self.level]

    def apply_overrides(self, **kwargs) -> "ECRConfig":
        """Apply non-None CLI overrides in place and return self."""
        for k, v in kwargs.items():
            if v is None:
                continue
            if k == "evidence_list":  # comma list -> toggle dict
                names = {n.strip() for n in v.split(",") if n.strip()}
                self.evidence = {e: (e in names or e == "semantic") for e in _DEFAULTS["evidence"]}
            elif k == "selector_type":
                self.selector = {**(self.selector or {}), "type": v}
            else:
                setattr(self, k, v)
        self._data.update({k: getattr(self, k) for k in self._data})
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self._data}


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base
