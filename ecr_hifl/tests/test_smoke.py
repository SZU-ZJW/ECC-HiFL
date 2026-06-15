"""Offline end-to-end smoke test for ECR-HiFL.

Runs on a small slice of the real Lite pools (no servers, no GPU) and asserts that, at every
level: candidates parse, evidence cards build with the expected groups, the rule selector and
baselines return valid predictions, metrics compute in [0,1], and oracle upper-bounds `first`
on recall. Also checks that the server-dependent selectors (llm) and evidence (verification)
import and degrade gracefully offline.

Run: ``./run_ecr.sh smoke``  (or ``python -m ecr_hifl.tests.test_smoke``).
"""

from __future__ import annotations

import sys
import traceback

from ..config import ECRConfig
from ..candidates import extract_all_responses, parse_candidates
from ..io_utils import load_jsonl_by_id, load_structure, load_problem_statements
from ..evidence.evidence_card import EvidenceBuilder
from ..selector.base import SelectionContext
from ..selector.baselines import make_baseline
from ..selector.rule_selector import RuleSelector
from ..selector.llm_selector import LLMSelector
from ..eval.metrics import compute_metrics, load_gold

LIMIT = 5


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_line_raw_bon_fallback() -> None:
    row = {
        "found_edit_locs": {"fallback.py": ["line: 1"]},
        "edit_loc_traj": {
            "all_responses": [[
                "```\npkg/a.py\nline: 10\n```",
                "```\npkg/b.py\nline: 20\n```",
            ]]
        },
    }
    raw = extract_all_responses(row, "line")
    _check(len(raw) == 2, "line all_responses was not flattened")
    _check("fallback.py" not in "\n".join(raw), "line parser used parsed fallback before raw BoN")
    cands = parse_candidates(row, "line", structure=None, sample=2)
    _check([c.files for c in cands] == [["pkg/a.py"], ["pkg/b.py"]],
           "line raw BoN responses did not parse as separate candidates")


def test_line_raw_response_fallback_for_empty_samples() -> None:
    row = {
        "found_files": ["pkg/a.py", "pkg/b.py"],
        "found_edit_locs": [{"pkg/a.py": [""], "pkg/b.py": [""]}],
        "edit_loc_traj": {
            "response": [
                "```full_path2/example.py\nmethod: target\nline: 20\n```",
                "```file1.py\nline: 7\n```",
                "```full_path_to_pkg_module/pkg_alias.py\nline: 8\n```",
            ],
        },
    }
    raw = extract_all_responses(row, "line")
    _check(len(raw) == 3 and "full_path2" in raw[0],
           "line parser did not fall back to raw response for empty parsed samples")
    cands = parse_candidates(row, "line", structure=None, sample=3)
    _check(cands[0].files == ["pkg/b.py"], "full_path2 placeholder was not mapped to found_files[1]")
    _check(cands[0].elements["pkg/b.py"] == ["function: target", "line: 20"],
           "raw line response was not normalized into function/line elements")
    _check(cands[1].files == ["pkg/a.py"], "file1.py placeholder was not mapped to found_files[0]")
    _check(cands[2].files == ["pkg/a.py"], "full_path_to_* placeholder was not mapped to found_files[0]")


def smoke_level(cfg: ECRConfig, level: str, ps: dict) -> None:
    pool = load_jsonl_by_id(cfg.pool_path(level))
    gold = load_gold(level, cfg.gold_path(level))
    iids = [i for i in pool if i in gold][:LIMIT]
    _check(iids, f"[{level}] no overlapping instances between pool and gold")

    builder = EvidenceBuilder.from_config(cfg)
    rule = RuleSelector(cfg.weights)
    llm = LLMSelector(cfg)  # offline -> must fall back to rule

    rule_preds, first_preds, oracle_preds = {}, {}, {}
    for iid in iids:
        structure = load_structure(iid, cfg.project_file_loc)
        _check(structure is not None, f"[{level}] missing structure for {iid}")
        cands = parse_candidates(pool[iid], level, structure, sample=cfg.sample)
        _check(len(cands) > 0, f"[{level}] no candidates parsed for {iid}")
        ctx = SelectionContext(instance_id=iid, level=level, problem_statement=ps.get(iid, ""),
                               structure=structure, gold=gold.get(iid))
        ctx.cards = builder.build_cards(cands, ctx)
        _check(len(ctx.cards) == len(cands), f"[{level}] cards/candidate length mismatch")
        _check("semantic" in ctx.cards[0].evidence, f"[{level}] semantic evidence missing")
        for grp in cfg.enabled_evidence():
            _check(grp in ctx.cards[0].evidence, f"[{level}] evidence group {grp} not built")

        r = rule.predict(cands, ctx)
        _check(r.index is not None and 0 <= r.index < len(cands), f"[{level}] rule index invalid")
        _check(len(r.scores) == len(cands), f"[{level}] rule scores length mismatch")
        rule_preds[iid] = r.fields

        l = llm.predict(cands, ctx)  # offline: should be a graceful fallback, still valid
        _check(l.fields is not None, f"[{level}] llm selector produced no fields")

        first_preds[iid] = make_baseline("first").predict(cands, ctx).fields
        oracle_preds[iid] = make_baseline("oracle").predict(cands, ctx).fields

    mr = compute_metrics(level, rule_preds, gold)
    mf = compute_metrics(level, first_preds, gold)
    mo = compute_metrics(level, oracle_preds, gold)
    for m in (mr, mf, mo):
        for key in ("precision", "recall", "f1"):
            _check(0.0 <= m[key] <= 1.0, f"[{level}] metric {key} out of range: {m[key]}")
    _check(mo["recall"] + 1e-9 >= mf["recall"], f"[{level}] oracle recall < first recall")
    print(f"  [{level}] OK  n={mr['n']}  first R={mf['recall']:.3f}  rule R={mr['recall']:.3f} "
          f"F1={mr['f1']:.3f}  oracle R={mo['recall']:.3f}")


def main() -> int:
    cfg = ECRConfig()  # offline Lite defaults
    cfg.evidence = {"semantic": True, "graph": True, "summary": True, "history": False, "verification": False}
    try:
        ps = load_problem_statements(cfg.dataset, cfg.split)
    except Exception as exc:
        print(f"  [info] problem statements unavailable offline ({type(exc).__name__}); using empty issue text")
        ps = {}
    print("ECR-HiFL smoke test (offline, 5 instances/level)")
    try:
        test_line_raw_bon_fallback()
        test_line_raw_response_fallback_for_empty_samples()
        for level in ("file", "function", "line"):
            smoke_level(cfg, level, ps)
    except Exception:
        print("SMOKE FAILED:")
        traceback.print_exc()
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
