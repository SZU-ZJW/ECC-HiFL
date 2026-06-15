#!/bin/bash
# [HiLoRM] 由 GRM4/STEP3/merge_zuhe.sh 迁移而来。
# 环境 (PROJECT_FILE_LOC / PYTHONPATH / OPENAI_API_KEY / HILORM_STAGE / 数据根变量)
# 由顶层 run.sh 设置。请用:  ./run.sh step3_retrieve batch merge.sh
# 数据根变量 (${PRED_LIST_ROOT} 等) 来自 config.yaml 的 paths.* ，运行前请核对。

SAMPLE=(
    3
    7
    10
    15
    30
)
for sample_num in ${SAMPLE[@]}; do
    for R in {1..1..1}; do
        PRED=${START_FILE_ROOT}/STEP1/results/yic9-${sample_num}/file_level/loc_outputs.jsonl
        RETR=${START_FILE_ROOT}/STEP3/result-7-yic9/embedding_level/yic9-${sample_num}/retrievel_embedding/combined_locs.jsonl
        if [ -f ${RETR} ] && [ -f ${PRED} ]; then
            python -m agentless.fl.combine \
                --retrieval_loc_file ${RETR} \
                --model_loc_file ${PRED} \
                --top_n 3 \
                --output_folder result-7-yic9/combine/yic9-${sample_num}/file_level_combined
        fi
    done
done
