#!/bin/bash
# [HiLoRM] 由 GRM4/STEP1/1-YiC9.sh 迁移而来。
# 环境 (PROJECT_FILE_LOC / PYTHONPATH / OPENAI_API_KEY / HILORM_STAGE / 数据根变量)
# 由顶层 run.sh 设置。请用:  ./run.sh step1_file batch yic9.sh
# 数据根变量 (${PRED_LIST_ROOT} 等) 来自 config.yaml 的 paths.* ，运行前请核对。

SAMPLE=(
    30
)
for sample_num in ${SAMPLE[@]}; do
    for T in {1..5..1}; do
        echo "Running related elements for Num_sample: ${sample_num} and Thread: ${T}"
        python -m agentless.fl.localize \
            --model q7bc \
            --file_level \
            --output_folder results/yic9-${sample_num}/${T}/file_level \
            --num_threads 10 \
            --sample ${sample_num} \
            --skip_existing \
            --pred_list ${PRED_LIST_ROOT}/swe-bench-lite/file_level/loc_outputs.jsonl;
    done
done
