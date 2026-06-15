import json
import os
import sys
import argparse
from tqdm import tqdm

def load_jsonl(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        sys.exit(1)
    
    file_data = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    instance_id = data['instance_id']
                    found_file = data["found_files"]
                    file_data[instance_id] = found_file
                except json.JSONDecodeError as e:
                    print(f"警告: 无法解析JSON行: {line}. 错误: {e}")
                except KeyError as e:
                    print(f"警告: 缺少必要的键: {e}")
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时出错: {e}")
        sys.exit(1)
    
    return file_data

def main(args):
    truth_file = "data/legacy/GRM4/others/swe-bench-lite/retrievel_embedding/retrieve_locs.jsonl"
    truth_data = load_jsonl(truth_file)

    # 文件路径配置
    save_file = os.path.join(args.save_folder, "combined_locs.jsonl")
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)
    
    pred_data = load_jsonl(args.pred_file)

    final_result = []

    for instance_id in tqdm(truth_data.keys(), desc="处理实例"):
        truth_files = truth_data[instance_id]
        if instance_id not in pred_data:
            print(f"警告: 实例ID {instance_id} 在预测数据中不存在")
            continue
        
        pred_files = pred_data[instance_id]
        instance_rank_result = []
        
        for truth_file in truth_files:
            if truth_file in pred_files:
                instance_rank_result.append(truth_file)
        
        meta_data = {
            "instance_id": instance_id,
            "found_files": instance_rank_result[:100],
        }
        final_result.append(meta_data)

    with open(save_file, "w") as file:
        for result in final_result:
            file.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    
    args = parser.parse_args()

    if os.path.exists(args.pred_file):
        main(args)