import json
import os
# 参考文件路径（只保留这些instance_id）
REF_FILE_PATH = 'data/legacy/GRM4-V/others/modified_files_Verified.jsonl'  # 请替换为你的参考文件路径

def get_instance_id_set(ref_file_path):
    id_set = set()
    with open(ref_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = data.get('instance_id')
            if instance_id is not None:
                id_set.add(instance_id)
    return id_set

def filter_main_file(main_file_path, valid_ids):
    seen_ids = set()
    filtered_lines = []
    with open(main_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = data.get('instance_id')
            if instance_id is None or instance_id not in valid_ids:
                continue
            if instance_id not in seen_ids:
                seen_ids.add(instance_id)
                filtered_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
    # 覆盖原文件
    with open(main_file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(filtered_lines)

if __name__ == '__main__':
    
    # 主文件路径
    sample_num_list = [3,7]
    for sample_num in sample_num_list:
        for i in range(1, 26):
            MAIN_FILE_PATH = f'./results/q7-{sample_num}/{i}/file_level/loc_outputs.jsonl'
            if os.path.exists(MAIN_FILE_PATH):
                valid_ids = get_instance_id_set(REF_FILE_PATH)
                filter_main_file(MAIN_FILE_PATH, valid_ids)
