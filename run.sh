#!/bin/bash
# =============================================================================
# HiLoRM 统一入口
#
#   ./run.sh <stage> <command> [args...]
#
# <stage>:  step1_file | step2_irrelevant | step3_retrieve | step5_element | step6_line
#
# <command>:
#   localize [args...]   ->  python agentless/fl/localize.py [args...]
#   retrieve [args...]   ->  python agentless/fl/retrieve.py [args...]   (step3)
#   combine  [args...]   ->  python agentless/fl/combine.py  [args...]   (step3)
#   batch [name] [args]  ->  运行 stages/<stage>/scripts/<name> 批跑脚本 (无 name 则列出)
#   py <file> [args...]  ->  python <file> [args...]   (跑 eval 等脚本的逃生口)
#   env                  ->  打印计算出的环境并退出 (调试用)
#   shell                ->  进入一个已设好环境的子 shell
#
# 该脚本统一设置: OPENAI_API_KEY / HILORM_STAGE / HILORM_CONFIG / HILORM_ROOT /
# PROJECT_FILE_LOC / PRED_LIST_ROOT / START_FILE_ROOT / EMBEDDING_PERSIST_DIR /
# PYTHONPATH(=stages/<stage>:core)，然后 cd 进 stages/<stage> 派发命令。
#
# Python interpreter: defaults to python; override with HILORM_PYTHON=/path/to/python.
# 配置文件: 默认仓库根 config.yaml;  可用  HILORM_CONFIG=/path/to.yaml  覆盖。
# 单实例示例:
#   ./run.sh step1_file localize --target_id django__django-10914 --sample 3 \
#            --output_folder results/smoke/file_level --num_threads 4 --skip_existing \
#            --pred_list "$PRED_LIST_ROOT/swe-bench-lite/file_level/loc_outputs.jsonl"
# =============================================================================
set -u

HILORM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HILORM_ROOT

VALID_STAGES="step1_file step2_irrelevant step3_retrieve step5_element step6_line"

usage() {
    sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    echo
    echo "可用 stage: $VALID_STAGES"
    exit "${1:-0}"
}

[ $# -lt 1 ] && usage 1
STAGE="$1"; shift
case "$STAGE" in
    -h|--help|help) usage 0 ;;
esac
if ! printf '%s\n' $VALID_STAGES | grep -qx "$STAGE"; then
    echo "ERROR: 未知 stage '$STAGE'。可用: $VALID_STAGES" >&2
    exit 1
fi

# --- Python interpreter ---
HILORM_PYTHON="${HILORM_PYTHON:-python}"
export HILORM_PYTHON

# --- 配置 + 端点环境 ---
export HILORM_STAGE="$STAGE"
export HILORM_CONFIG="${HILORM_CONFIG:-$HILORM_ROOT/config.yaml}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-agent123}"
if [ ! -f "$HILORM_CONFIG" ]; then
    echo "ERROR: 找不到 config: $HILORM_CONFIG" >&2; exit 1
fi

# 从 config.yaml 的 paths.* 导出数据根变量
_cfg_exports="$("$HILORM_PYTHON" - "$HILORM_CONFIG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
p = cfg.get("paths", {}) or {}
for env_name, key in [("PROJECT_FILE_LOC", "project_file_loc"),
                      ("PRED_LIST_ROOT", "pred_list_root"),
                      ("START_FILE_ROOT", "start_file_root"),
                      ("EMBEDDING_PERSIST_DIR", "embedding_persist_dir")]:
    val = p.get(key)
    if val:
        print(f'export {env_name}="{val}"')
PYEOF
)"
eval "$_cfg_exports"

# --- PYTHONPATH: 分层 namespace 合并 (stage 在前, core 在后) ---
export PYTHONPATH="$HILORM_ROOT/stages/$STAGE:$HILORM_ROOT/core${PYTHONPATH:+:$PYTHONPATH}"

STAGE_DIR="$HILORM_ROOT/stages/$STAGE"
cd "$STAGE_DIR" || { echo "ERROR: 无法进入 $STAGE_DIR" >&2; exit 1; }

CMD="${1:-}"; [ $# -ge 1 ] && shift

case "$CMD" in
    # 用 -m 模块式调用: 经分层 PYTHONPATH 解析, 无论模块在 stages/ 还是 core/ 都命中
    localize) exec "$HILORM_PYTHON" -m agentless.fl.localize "$@" ;;
    retrieve) exec "$HILORM_PYTHON" -m agentless.fl.retrieve "$@" ;;
    combine)  exec "$HILORM_PYTHON" -m agentless.fl.combine  "$@" ;;
    py)       exec "$HILORM_PYTHON" "$@" ;;
    batch)
        name="${1:-}"; [ $# -ge 1 ] && shift
        if [ -z "$name" ]; then
            echo "stages/$STAGE/scripts/ 下可用批跑脚本:"
            ls -1 "$STAGE_DIR/scripts" 2>/dev/null | sed 's/^/  /'
            exit 0
        fi
        script="$STAGE_DIR/scripts/$name"
        [ -f "$script" ] || { echo "ERROR: 找不到批跑脚本 $script" >&2; exit 1; }
        exec bash "$script" "$@"
        ;;
    env)
        echo "HILORM_ROOT=$HILORM_ROOT"
        echo "HILORM_STAGE=$HILORM_STAGE"
        echo "HILORM_CONFIG=$HILORM_CONFIG"
        echo "HILORM_PYTHON=$HILORM_PYTHON"
        echo "OPENAI_API_KEY=$OPENAI_API_KEY"
        echo "PROJECT_FILE_LOC=${PROJECT_FILE_LOC:-}"
        echo "PRED_LIST_ROOT=${PRED_LIST_ROOT:-}"
        echo "START_FILE_ROOT=${START_FILE_ROOT:-}"
        echo "EMBEDDING_PERSIST_DIR=${EMBEDDING_PERSIST_DIR:-}"
        echo "PYTHONPATH=$PYTHONPATH"
        echo "cwd=$(pwd)"
        ;;
    shell) exec bash ;;
    ""|-h|--help|help) usage 0 ;;
    *) echo "ERROR: 未知 command '$CMD'" >&2; usage 1 ;;
esac
