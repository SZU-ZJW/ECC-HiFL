#!/bin/bash
# [HiLoRM] 由 GRM4/STEP3/run-embedding.sh 迁移而来。
# 环境 (PROJECT_FILE_LOC / PYTHONPATH / OPENAI_API_KEY / HILORM_STAGE / 数据根变量)
# 由顶层 run.sh 设置。请用:  ./run.sh step3_retrieve batch rank_embedding.sh
# 数据根变量 (${PRED_LIST_ROOT} 等) 来自 config.yaml 的 paths.* ，运行前请核对。

SAMPLE=(
    3
    7
    10
    15
    30
)
for sample_num in ${SAMPLE[@]}; do
    for T in {1..1..1}; do
        PREFILE=${START_FILE_ROOT}/STEP2/results/yic9-${sample_num}/file_level_irrelevant/loc_outputs.jsonl
        if [ -f "$PREFILE" ]; then
            python "${HILORM_ROOT}/eval/rank_pred.py" --pred_file $PREFILE \
                            --save_folder result-7-yic9/embedding_level/yic9-${sample_num}/retrievel_embedding;
        fi
    done
done
