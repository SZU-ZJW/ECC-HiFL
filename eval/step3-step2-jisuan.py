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
        tuple: (precision, recall, f1, is_fully_covered, is_exact_match, top1_hit, top3_hit, top5_hit)
    """
    if not modified_files:  # 如果没有实际修改的文件
        return 0, 0, 0, True, True, True, True, True
        
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
    
    # 计算TOP-K指标
    # 检查前K个预测文件中是否包含至少一个正确文件
    top1_hit = len(set(found_files[:1]).intersection(actual)) > 0 if len(found_files) >= 1 else False
    top3_hit = len(set(found_files[:3]).intersection(actual)) > 0 if len(found_files) >= 1 else False
    top5_hit = len(set(found_files[:5]).intersection(actual)) > 0 if len(found_files) >= 1 else False
    
    return precision, recall, f1, is_fully_covered, is_exact_match, top1_hit, top3_hit, top5_hit

def process_experiment(modified_data, experiment_data, experiment_name):
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
            precision, recall, f1, is_fully_covered, is_exact_match, top1_hit, top3_hit, top5_hit = calculate_coverage(modified_files, found_files)
            stats['precision'] += precision
            stats['recall'] += recall
            stats['f1'] += f1
            
            if is_fully_covered:
                fully_covered_count += 1
            if is_exact_match:
                exact_match_count += 1
            if top1_hit:
                top1_count += 1
            if top3_hit:
                top3_count += 1
            if top5_hit:
                top5_count += 1
    
    if count > 0:
        print(f"\n{experiment_name} Results (Total: {count}):")
        print(f"Average Precision: {stats['precision'] / count:.4f}")
        print(f"Average Recall: {stats['recall'] / count:.4f}")
        print(f"Average F1: {stats['f1'] / count:.4f}")
        print(f"Full Coverage Rate: {fully_covered_count/count*100:.2f}% ({fully_covered_count}/{count})")
        print(f"Exact Match Rate: {exact_match_count/count*100:.2f}% ({exact_match_count}/{count})")
        print(f"TOP1 Hit Rate: {top1_count/count*100:.2f}% ({top1_count}/{count})")
        print(f"TOP3 Hit Rate: {top3_count/count*100:.2f}% ({top3_count}/{count})")
        print(f"TOP5 Hit Rate: {top5_count/count*100:.2f}% ({top5_count}/{count})")

def main():
    sample_num_list = [3,7,10,15,30]
    # 文件路径
    modified_files_path = 'data/legacy/GRM4/others/modified_files.jsonl'
    modified_data = load_jsonl_file(modified_files_path)
    for sample_num in sample_num_list:
        ir_path = f'result-7-yic9/combine/yic9-{sample_num}/file_level_combined/combined_locs.jsonl'
        if os.path.exists(ir_path):
            print(f"==============={sample_num}===============")
            ir_data = load_jsonl_file(ir_path, None)
            process_experiment(modified_data, ir_data, "Irrelevant Result") 

if __name__ == "__main__":
    main()
