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

def calculate_topk_accuracy(modified_files, found_files, k):
    """计算TOP-K准确率
    
    Args:
        modified_files: 实际修改的文件列表
        found_files: 预测找到的相关文件列表（按相关性排序）
        k: TOP-K中的K值
        
    Returns:
        float: TOP-K准确率
    """
    if not modified_files:
        return 1.0  # 如果没有实际修改的文件，认为是正确的
    
    # 取前K个预测文件
    top_k_predicted = set(found_files[:k])
    actual = set(modified_files)
    
    # 计算交集
    intersection = actual.intersection(top_k_predicted)
    
    # TOP-K准确率：至少有一个真实文件在前K个预测中
    return 1.0 if len(intersection) > 0 else 0.0

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

def process_experiment(modified_data, experiment_data, string, experiment_name):
    """处理单个实验的数据"""
    stats = defaultdict(float)
    count = 0
    fully_covered_count = 0
    exact_match_count = 0
    top1_count = 0
    top3_count = 0
    top5_count = 0
    
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
            
            # 计算TOP-K指标
            if calculate_topk_accuracy(modified_files, found_files, 1):
                top1_count += 1
            if calculate_topk_accuracy(modified_files, found_files, 3):
                top3_count += 1
            if calculate_topk_accuracy(modified_files, found_files, 5):
                top5_count += 1
    
    if count > 0:
        if stats['recall'] / count > 0.1:
            print(string)
            #print(f"\n{experiment_name} Results (Total: {count}):")
            print(f"Prec: {stats['precision'] / count:.4f}")
            print(f"Recall: {stats['recall'] / count:.4f}")
            print(f"F1: {stats['f1'] / count:.4f}")
            # print(f"TOP1: {top1_count/count:.4f} ({top1_count}/{count})")
            # print(f"TOP3: {top3_count/count:.4f} ({top3_count}/{count})")
            # print(f"TOP5: {top5_count/count:.4f} ({top5_count}/{count})")
            #print(f"Full Coverage Rate: {fully_covered_count/count*100:.2f}% ({fully_covered_count}/{count})")
            print(f"EM: ({exact_match_count}/{count})")
            print("\n")
    return stats['recall'] / count

def main():
    sample_num_list = [3,7,10,15,30]

    modified_files_path = 'data/legacy/GRM4/others/modified_files.jsonl'
    
    for sample_num in sample_num_list:
        recall = 0
        print("="*15 + f" {sample_num} " + "="*15)
        for i in range(1, 2):
            file_path = f"./results/yirm-{sample_num}/file_level/loc_outputs.jsonl"
            save_path = "./file_loc.jsonl"
            if os.path.exists(file_path):
                string = "-"*10 + f" {sample_num}-{i} "+ "-"*10
                with open(save_path, "w") as f:
                    with open(file_path, "r") as f2:
                        for line in f2:
                            data = json.loads(line)
                            instance_id = data["instance_id"]
                            found_files = data["found_files"]
                            metadata = {
                                "instance_id": instance_id,
                                "found_files": found_files
                            }
                            f.write(json.dumps(metadata) + "\n")

                modified_data = load_jsonl_file(modified_files_path)
                step1_data = load_jsonl_file(save_path, None)

                recall_now = process_experiment(modified_data, step1_data, string, "Step1 Result")
                if recall_now > recall:
                    recall = recall_now
                    best_i = i
        print(f"Best Recall: {recall:.4f} at {best_i}")

if __name__ == "__main__":
    main()
