#!/bin/bash
# [HiLoRM] 由 GRM4/STEP5/5-loc_related.sh 迁移而来。
# 环境 (PROJECT_FILE_LOC / PYTHONPATH / OPENAI_API_KEY / HILORM_STAGE / 数据根变量)
# 由顶层 run.sh 设置。请用:  ./run.sh step5_element batch related.sh
# 数据根变量 (${PRED_LIST_ROOT} 等) 来自 config.yaml 的 paths.* ，运行前请核对。

SAMPLE=(
    15
)
for sample_num in ${SAMPLE[@]}; do
    for T in {11..15..1}; do
        echo "Running related elements for Num_sample: ${sample_num} at NO.${T}"
        python -m agentless.fl.localize --related_level \
            --model q7bc \
            --output_folder results/grm7-${sample_num}/${T}/related_elements \
            --top_n 3 \
            --compress_assign \
            --compress \
            --start_file ${START_FILE_ROOT}/STEP3/result-7-grm/combine/grm-${sample_num}/file_level_combined/combined_locs.jsonl \
            --num_threads 20 \
            --skip_existing \
            --sample ${sample_num} \
            --pred_list ${PRED_LIST_ROOT}/GRM-${sample_num}/related_elements/loc_outputs.jsonl;
    done
done
