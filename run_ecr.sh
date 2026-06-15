#!/bin/bash
# =============================================================================
# ECR-HiFL entry point (mirrors run.sh's env wiring; does not touch the stages).
#
#   ./run_ecr.sh <command> [args...]
#
# <command>:
#   select   [args]   -> python -m ecr_hifl.pipeline      (run one selector, write results)
#   baseline [args]   -> python -m ecr_hifl.experiments baseline   (first/random/majority/oracle/rule table)
#   ablation [args]   -> python -m ecr_hifl.experiments ablation   (evidence x sample_nums grid)
#   eval     [args]   -> python -m ecr_hifl.eval.evaluate  (score a results/pool jsonl vs gold)
#   smoke             -> python -m ecr_hifl.tests.test_smoke        (offline end-to-end smoke)
#   precompute [args] -> python -m ecr_hifl.precompute  (cache graphs / evidence cards)
#   py <file> [args]  -> python <file> [args]   (escape hatch)
#   env               -> print computed environment
#
# Sets PYTHONPATH(=core:<repo>) so the reused HiLoRM core + the ecr_hifl package both import,
# plus HILORM_STAGE/HILORM_CONFIG/OPENAI_API_KEY so EP (model endpoints) loads when servers are up.
# Python interpreter: defaults to python; override with HILORM_PYTHON.
# ECR config: pass --config configs/ecr_*.yaml to the subcommands (defaults baked into ecr_hifl.config).
#
# Examples:
#   ./run_ecr.sh baseline --config configs/ecr_lite.yaml --level file
#   ./run_ecr.sh select   --config configs/ecr_lite.yaml --level file --selector rule --eval
#   ./run_ecr.sh ablation --config configs/ecr_ablation.yaml --limit 50
#   ./run_ecr.sh precompute graphs --config configs/ecr_lite.yaml --levels file,function,line
#   ./run_ecr.sh smoke
# =============================================================================
set -u

HILORM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HILORM_ROOT

HILORM_PYTHON="${HILORM_PYTHON:-python}"
export HILORM_PYTHON

export HILORM_STAGE="${HILORM_STAGE:-step1_file}"
export HILORM_CONFIG="${HILORM_CONFIG:-$HILORM_ROOT/config.yaml}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-agent123}"
# PYTHONPATH: core (shared agentless utils) + repo root (ecr_hifl package)
export PYTHONPATH="$HILORM_ROOT/core:$HILORM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$HILORM_ROOT" || exit 1
CMD="${1:-}"; [ $# -ge 1 ] && shift

case "$CMD" in
    select)   exec "$HILORM_PYTHON" -m ecr_hifl.pipeline "$@" ;;
    baseline) exec "$HILORM_PYTHON" -m ecr_hifl.experiments baseline "$@" ;;
    ablation) exec "$HILORM_PYTHON" -m ecr_hifl.experiments ablation "$@" ;;
    eval)     exec "$HILORM_PYTHON" -m ecr_hifl.eval.evaluate "$@" ;;
    smoke)    exec "$HILORM_PYTHON" -m ecr_hifl.tests.test_smoke "$@" ;;
    precompute) exec "$HILORM_PYTHON" -m ecr_hifl.precompute "$@" ;;
    py)       exec "$HILORM_PYTHON" "$@" ;;
    env)
        echo "HILORM_ROOT=$HILORM_ROOT"
        echo "HILORM_PYTHON=$HILORM_PYTHON"
        echo "HILORM_STAGE=$HILORM_STAGE"
        echo "HILORM_CONFIG=$HILORM_CONFIG"
        echo "PYTHONPATH=$PYTHONPATH"
        echo "cwd=$(pwd)"
        ;;
    ""|-h|--help|help)
        sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        ;;
    *) echo "ERROR: unknown command '$CMD'" >&2; exit 1 ;;
esac
