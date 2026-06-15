#!/bin/bash
# =============================================================================
# Resolved-rate (修复解决率) evaluation — SWE-bench harness, runnable from this repo.
#
#   ./run_resolved_eval.sh <predictions.jsonl> [run_id] [dataset] [max_workers]
#
# Runs the copied SWE-bench harness in ./swebench_eval. Use an environment with
# swebench dependencies and Docker access.
# Requires Docker. The harness writes its report `<model_name_or_path>.<run_id>.json` and
# per-instance logs under ./swebench_eval/.
#
#   predictions.jsonl : lines of {instance_id, model_patch, model_name_or_path}
#   run_id            : defaults to the predictions filename (no extension)
#   dataset           : defaults to princeton-nlp/SWE-bench_Lite
#   max_workers       : defaults to 8
#
# Override the interpreter with EVAL_PYTHON=/path/to/python.
# Example:
#   EVAL_PYTHON=/path/to/eval-env/bin/python ./run_resolved_eval.sh predictions.jsonl run-id
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$ROOT/swebench_eval"

EVAL_PYTHON="${EVAL_PYTHON:-python}"
if ! command -v "$EVAL_PYTHON" >/dev/null 2>&1 && [ ! -x "$EVAL_PYTHON" ]; then
    echo "ERROR: eval python not found at $EVAL_PYTHON (set EVAL_PYTHON=...)." >&2; exit 1
fi
[ -d "$HARNESS_DIR/swebench" ] || { echo "ERROR: harness not found at $HARNESS_DIR/swebench" >&2; exit 1; }

PRED="${1:-}"; [ -z "$PRED" ] && { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }
[ -f "$PRED" ] || { echo "ERROR: predictions file not found: $PRED" >&2; exit 1; }
PRED="$(cd "$(dirname "$PRED")" && pwd)/$(basename "$PRED")"   # absolutize (we cd below)
RUN_ID="${2:-$(basename "$PRED" | sed 's/\.[^.]*$//')}"
DATASET="${3:-princeton-nlp/SWE-bench_Lite}"
MAX_WORKERS="${4:-8}"

command -v docker >/dev/null 2>&1 || echo "WARNING: docker not on PATH — the harness needs it." >&2

echo "== resolved-rate eval =="
echo "  python      = $EVAL_PYTHON"
echo "  harness     = $HARNESS_DIR (swebench via PYTHONPATH)"
echo "  predictions = $PRED"
echo "  run_id      = $RUN_ID   dataset = $DATASET   max_workers = $MAX_WORKERS"

cd "$HARNESS_DIR" || exit 1   # harness writes report + logs in cwd
PYTHONPATH="$HARNESS_DIR" "$EVAL_PYTHON" -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET" --predictions_path "$PRED" \
    --max_workers "$MAX_WORKERS" --run_id "$RUN_ID"
rc=$?

# summarize the resolved rate from the report the harness just wrote
report="$(ls -t "$HARNESS_DIR"/*."$RUN_ID".json 2>/dev/null | head -1)"
if [ -n "$report" ]; then
    echo "== report: $report =="
    "$EVAL_PYTHON" - "$report" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
tot = r.get("total_instances"); res = r.get("resolved_instances")
comp = r.get("completed_instances"); sub = r.get("submitted_instances")
print(f"  submitted={sub} completed={comp} resolved={res} total={tot}")
if tot and res is not None:
    print(f"  %% Resolved = {100.0*res/tot:.2f}%  ({res}/{tot})")
PYEOF
fi
exit $rc
