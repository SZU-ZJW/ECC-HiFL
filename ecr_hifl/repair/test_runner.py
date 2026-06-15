"""Run SWE-bench tests on generated patches → resolved rate (Docker / harness-gated).

Drives the self-contained harness copied into ``<repo>/swebench_eval`` using the **eval** conda
env (`<eval-env>`, NOT asgl) — it has swebench + docker. Expects a predictions
jsonl in SWE-bench format (``instance_id``, ``model_patch``, ``model_name_or_path``) and shells out
to ``python -m swebench.harness.run_evaluation`` (via ``PYTHONPATH=swebench_eval``). The thin
shell wrapper ``run_resolved_eval.sh`` does the same from the CLI.

This is the ``% Resolved`` evaluation hook. Offline / no-Docker it prints the exact command
to run instead of failing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional

# repo root = three levels up from ecr_hifl/repair/test_runner.py
_REPO_ROOT = os.environ.get("HILORM_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_HARNESS_DIR = os.path.join(_REPO_ROOT, "swebench_eval")
_EVAL_PYTHON = os.environ.get("EVAL_PYTHON", "python")


def have_harness() -> bool:
    python_ok = os.path.exists(_EVAL_PYTHON) or shutil.which(_EVAL_PYTHON) is not None
    return (os.path.isdir(os.path.join(_HARNESS_DIR, "swebench"))
            and python_ok
            and shutil.which("docker") is not None)


def run_evaluation(predictions_path: str, dataset: str = "princeton-nlp/SWE-bench_Lite",
                   run_id: str = "ecr_hifl", max_workers: int = 8) -> Optional[str]:
    """Run the harness on a predictions jsonl; return the report path, or None if prerequisites missing."""
    predictions_path = os.path.abspath(predictions_path)
    cmd = [_EVAL_PYTHON, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", dataset, "--predictions_path", predictions_path,
           "--max_workers", str(max_workers), "--run_id", run_id]
    if not have_harness():
        print("[repair] eval env / harness / Docker unavailable. Run manually when ready:\n  "
              f"EVAL_PYTHON={_EVAL_PYTHON} PYTHONPATH={_HARNESS_DIR} (cd {_HARNESS_DIR} && "
              + " ".join(cmd) + ")\n  or: ./run_resolved_eval.sh " + predictions_path + f" {run_id}")
        return None
    env = {**os.environ, "PYTHONPATH": _HARNESS_DIR}
    subprocess.run(cmd, cwd=_HARNESS_DIR, env=env, check=False)
    # harness writes <model_name_or_path>.<run_id>.json in cwd
    candidates = [f for f in os.listdir(_HARNESS_DIR) if f.endswith(f".{run_id}.json")]
    if not candidates:
        return None
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(_HARNESS_DIR, f)), reverse=True)
    return os.path.join(_HARNESS_DIR, candidates[0])


def resolved_rate(report_path: str) -> Optional[float]:
    if not report_path or not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        report = json.load(f)
    total = report.get("total_instances") or report.get("submitted_instances")
    resolved = report.get("resolved_instances")
    if not total or resolved is None:
        return None
    return resolved / total
