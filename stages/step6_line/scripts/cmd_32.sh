#!/bin/bash
# [HiLoRM] 由 GRM4/STEP6/6-32-cmd.sh 迁移而来。
# 环境 (PROJECT_FILE_LOC / PYTHONPATH / OPENAI_API_KEY / HILORM_STAGE / 数据根变量)
# 由顶层 run.sh 设置。请用:  ./run.sh step6_line batch cmd_32.sh
# 数据根变量 (${PRED_LIST_ROOT} 等) 来自 config.yaml 的 paths.* ，运行前请核对。

SAMPLE=(
    10
)
for sample_num in ${SAMPLE[@]}; do
    for T in {1..10..1}; do
        echo "Running HiLoRM-${sample_num}-sampling No.${T}"
        python -m agentless.fl.localize \
            --fine_grain_line_level \
            --model q7bc \
            --output_folder results/GRM7-${sample_num}/${T}/edit_location_samples \
            --top_n 3 \
            --compress \
            --temperature 1 \
            --num_samples 1 \
            --start_file ${START_FILE_ROOT}/STEP5/results/grm7-${sample_num}/related_elements/loc_outputs.jsonl \
            --num_threads 10 \
            --skip_existing \
            --sample ${sample_num} \
            --pred_list ${PRED_LIST_ROOT}/GRM-${sample_num}/edit_location_samples/loc_outputs.jsonl;
    done
done
