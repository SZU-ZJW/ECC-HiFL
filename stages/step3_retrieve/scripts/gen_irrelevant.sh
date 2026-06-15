#!/bin/bash
# [HiLoRM] 由 GRM4/STEP3/2-irrelevant.sh 迁移而来。
# 环境 (PROJECT_FILE_LOC / PYTHONPATH / OPENAI_API_KEY / HILORM_STAGE / 数据根变量)
# 由顶层 run.sh 设置。请用:  ./run.sh step3_retrieve batch gen_irrelevant.sh
# 数据根变量 (${PRED_LIST_ROOT} 等) 来自 config.yaml 的 paths.* ，运行前请核对。

SAMPLE=(
    3
    7
    10
    15
    30
)
for sample_num in ${SAMPLE[@]}; do
    for T in {21..30..1}; do
        python -m agentless.fl.localize --model q7bc --file_level --irrelevant --output_folder results/swe-bench-lite-ir-${sample_num}-F-1@${T}/file_level_irrelevant --num_threads 5 --sample ${sample_num} --skip_existing --pred_list ${PRED_LIST_ROOT}/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl;
    done
done
