import json
import os
from collections import defaultdict

def read_instance_ids(file_path):
    """从JSONL文件中读取所有instance_id并返回列表
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        list: instance_id列表
    """
    instance_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            instance_ids.append(item['instance_id'])
    return instance_ids

def load_jsonl_file(file_path, instance_ids=None):
    """加载JSONL文件
    
    Args:
        file_path: JSONL文件路径
        instance_ids: 可选，要加载的instance_id列表。如果提供，则只加载这些id对应的数据
    
    Returns:
        dict: {instance_id: data}
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 如果提供了instance_ids列表，则只加载列表中的id
            if instance_ids is None or item['instance_id'] in instance_ids:
                data[item['instance_id']] = item
    return data

def calculate_coverage(modified_files, found_files):
    """计算覆盖率
    
    Args:
        modified_files: 实际修改的文件列表
        found_files: 预测找到的相关文件列表
        
    Returns:
        tuple: (precision, recall, f1, is_fully_covered, is_exact_match)
    """
    if not modified_files:  # 如果没有实际修改的文件
        return 0, 0, 0, True, True
        
    # 转换为集合
    predicted = set(found_files)
    actual = set(modified_files)
    
    # 检查是否完全覆盖和精准匹配
    is_fully_covered = actual.issubset(predicted)
    is_exact_match = actual == predicted
    
    # 计算真阳性（正确预测的文件）
    true_positives = len(actual.intersection(predicted))
    
    # 计算精确率和召回率
    precision = true_positives / len(predicted) if predicted else 0
    recall = true_positives / len(actual) if actual else 0
    
    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, is_fully_covered, is_exact_match

def process_experiment(modified_data, experiment_data, experiment_name):
    """处理单个实验的数据"""
    stats = defaultdict(float)
    count = 0
    fully_covered_count = 0
    exact_match_count = 0
    
    for instance_id, data in modified_data.items():
        if instance_id in experiment_data:
            modified_files = data.get('modified_files', [])
            found_files = experiment_data[instance_id].get('found_files', [])
            
            if not modified_files:
                continue
                
            count += 1
            precision, recall, f1, is_fully_covered, is_exact_match = calculate_coverage(modified_files, found_files)
            stats['precision'] += precision
            stats['recall'] += recall
            stats['f1'] += f1
            
            if is_fully_covered:
                fully_covered_count += 1
            if is_exact_match:
                exact_match_count += 1
    
    if count > 0:
        print(f"\n{experiment_name} Results (Total: {count}):")
        print(f"Precision: {stats['precision'] / count:.4f}")
        print(f"Recall: {stats['recall'] / count:.4f}")
        print(f"F1: {stats['f1'] / count:.4f}")
        print(f"Full Coverage Rate: {fully_covered_count/count*100:.2f}% ({fully_covered_count}/{count})")
        #print(f"Exact Match Rate: {exact_match_count/count*100:.2f}% ({exact_match_count}/{count})")
        print(f"EM: ({exact_match_count}/{count})")

def main():
    sample_num_list = [30]
    # 文件路径
    modified_files_path = 'data/modified_files.jsonl'
    modified_data = load_jsonl_file(modified_files_path)
    for sample_num in sample_num_list:
        for pred in range(1,2):
            print(f"="*30 + f"{pred}" + f"="*30)
            for retr in range(2,3):
                pred_path = f"results-q7bc/file_level/7b-{sample_num}/{pred}/file_level/loc_outputs.jsonl"
                irr_path = f"results-q7bc/irrelevant_level/Q7-{sample_num}/{retr}/file_level_irrelevant/loc_outputs.jsonl"
                combined_path = f"results-q7bc/combine/Q7-{sample_num}/P_{pred}-R_{retr}/file_level_combined/combined_locs.jsonl"
                if os.path.exists(combined_path) and os.path.exists(pred_path) and os.path.exists(irr_path):
                    print(f"\nProcessing SAMPLE {sample_num} with PRED {pred} and RETR {retr}...")
                    save_path = "./file_loc.jsonl"
                    with open(save_path, "w") as f:
                        with open(pred_path, "r") as f2:
                            for line in f2:
                                data = json.loads(line)
                                instance_id = data["instance_id"]
                                found_files = data["found_files"]
                                metadata = {
                                    "instance_id": instance_id,
                                    "found_files": found_files
                                }
                                f.write(json.dumps(metadata) + "\n")
                    step1_data = load_jsonl_file(save_path, None)
                    step2_data = load_jsonl_file(irr_path, None)
                    step3_data = load_jsonl_file(combined_path, None)

                    process_experiment(modified_data, step1_data, "Step1 Result") 
                    process_experiment(modified_data, step2_data, "Irrelevant Result") 
                    process_experiment(modified_data, step3_data, "Combined Result")
                    print(f"-"*10 + f"END" + f"-"*10) 

if __name__ == "__main__":
    main()
