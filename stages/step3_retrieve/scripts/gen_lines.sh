#!/bin/bash
# [HiLoRM] 由 GRM4/STEP3/6-localize_edit_location.sh 迁移而来。
# 环境 (PROJECT_FILE_LOC / PYTHONPATH / OPENAI_API_KEY / HILORM_STAGE / 数据根变量)
# 由顶层 run.sh 设置。请用:  ./run.sh step3_retrieve batch gen_lines.sh
# 数据根变量 (${PRED_LIST_ROOT} 等) 来自 config.yaml 的 paths.* ，运行前请核对。

python -m agentless.fl.localize --fine_grain_line_level --model q7bc --output_folder results/swe-bench-lite/edit_location_samples --top_n 3 --compress --temperature 0.8 --num_samples 4 --start_file results/swe-bench-lite/related_elements/loc_outputs.jsonl --num_threads 1 --skip_existing
