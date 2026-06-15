# swebench_eval — resolved-rate (修复解决率) evaluation harness

Self-contained copy of the SWE-bench evaluation **harness package** taken from
`data/swe-bench/eval` (swebench 2.1.2). Only the ~0.6 MB code (`swebench/` + setup files +
`patch_check.py`) was copied — **not** the ~700 MB of logs / data / Docker artifacts / prediction
sets in the source tree.

## How to run (from the repo root)

```bash
./run_resolved_eval.sh <predictions.jsonl> [run_id] [dataset] [max_workers]
```

- Uses the **`eval`** conda env (`<eval-env>`, **not** `asgl`) — it has
  `swebench` + `docker`. Override with `EVAL_PYTHON=/path/to/python`.
- Runs the harness via `PYTHONPATH=swebench_eval` so this copy is used **without** reinstalling /
  repointing the shared `eval` env's editable install.
- **Requires Docker** (present: 27.3.1). The harness writes `<model_name_or_path>.<run_id>.json`
  (resolved/total) and per-instance logs here under `swebench_eval/`.

Predictions are jsonl lines of `{instance_id, model_patch, model_name_or_path}`.

Also wired programmatically in `ecr_hifl/repair/test_runner.py`
(`run_evaluation(...)` / `resolved_rate(...)`).

## Where the patches come from

`% Resolved` needs patches. The ECR chain is: localize (`ecr_hifl.pipeline`) → generate patch
(`ecr_hifl.repair.patch_generator`, needs the generation server) → assemble a predictions jsonl →
`run_resolved_eval.sh`. Existing FL-method baseline prediction sets to compare against live at
`data/swe-bench/eval/ECR/*.jsonl` (ACR / CoSIL / FLEXFL / MHT-RD) and `.../data/`.
