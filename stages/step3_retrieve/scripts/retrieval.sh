#!/bin/bash
# [HiLoRM] 由 GRM4/STEP3/3-retrieval.sh 迁移而来。
# 环境 (PROJECT_FILE_LOC / PYTHONPATH / OPENAI_API_KEY / HILORM_STAGE / 数据根变量)
# 由顶层 run.sh 设置。请用:  ./run.sh step3_retrieve batch retrieval.sh
# 数据根变量 (${PRED_LIST_ROOT} 等) 来自 config.yaml 的 paths.* ，运行前请核对。

python -m agentless.fl.retrieve --index_type simple --filter_type given_files --filter_file results/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl --output_folder results/swe-bench-lite/retrievel_embedding --persist_dir  ${EMBEDDING_PERSIST_DIR} --num_threads 1
