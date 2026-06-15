import json
import re
import sys
from collections import defaultdict

def load_standard_results(file_path):
    """加载标准修改结果"""
    results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            instance_id = data["instance_id"]
            code_elements = {}
            for file_path, elements in data["result"]["code_elements"].items():
                code_elements[file_path] = []
                for element in elements:
                    elem_type = element["type"]
                    elem_value = element["value"]
                    if "start_line" in element and "end_line" in element:
                        code_elements[file_path].append((elem_type, elem_value, element["start_line"], element["end_line"]))
                    else:
                        code_elements[file_path].append((elem_type, elem_value, None, None))
            results[instance_id] = code_elements
    return results

def load_predictions(file_path):
    """加载预测结果"""
    predictions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            instance_id = data["instance_id"]
            code_elements = defaultdict(list)
            
            for file_path, elements in data["edit_loc"].items():
                # Skip empty elements
                if not elements or (len(elements) == 1 and elements[0] == ""):
                    continue
                
                for element in elements:
                    if not element:
                        continue
                    
                    # Function pattern
                    func_match = re.search(r"function:\s*(\w+(?:\.\w+)*)", element)
                    if func_match:
                        func_name = func_match.group(1)
                        code_elements[file_path].append(("function", func_name, None, None))
                    
                    # Class pattern
                    class_match = re.search(r"class:\s*(\w+(?:\.\w+)*)", element)
                    if class_match:
                        class_name = class_match.group(1)
                        code_elements[file_path].append(("class", class_name, None, None))
                    
                    # Line pattern (just record it as a line)
                    line_match = re.search(r"line:\s*(\d+)", element)
                    if line_match and not func_match and not class_match:
                        line_num = int(line_match.group(1))
                        code_elements[file_path].append(("line", str(line_num), line_num, line_num))
            
            predictions[instance_id] = dict(code_elements)
    
    return predictions

def normalize_path(path):
    """规范化文件路径以进行比较"""
    return path.split('/')[-1]

def evaluate(standard, predictions):
    """评估预测结果，计算precision, recall和F1分数"""
    total_tp = 0  # 真正例
    total_fp = 0  # 假正例
    total_fn = 0  # 假反例
    
    # 查看每个实例
    instance_metrics = {}
    
    for instance_id, std_elements in standard.items():
        if instance_id not in predictions:
            # 如果预测中不存在该实例，所有标准元素都是假反例
            fn = sum(len(elements) for elements in std_elements.values())
            total_fn += fn
            instance_metrics[instance_id] = {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": fn}
            continue
            
        pred_elements = predictions[instance_id]
        tp = 0  # 当前实例的真正例
        fp = 0  # 当前实例的假正例
        fn = 0  # 当前实例的假反例
        
        # 对每个文件路径
        for std_file, std_file_elements in std_elements.items():
            normalized_std_file = normalize_path(std_file)
            
            # 寻找预测中匹配的文件
            matching_pred_file = None
            for pred_file in pred_elements.keys():
                if normalize_path(pred_file) == normalized_std_file:
                    matching_pred_file = pred_file
                    break
            
            if matching_pred_file:
                # 找到匹配的文件，对比元素
                pred_file_elements = pred_elements[matching_pred_file]
                
                for std_elem in std_file_elements:
                    std_type, std_value, std_start, std_end = std_elem
                    found = False
                    
                    for pred_elem in pred_file_elements:
                        pred_type, pred_value, pred_start, pred_end = pred_elem
                        
                        # 匹配类型和值
                        if std_type == pred_type and std_value == pred_value:
                            found = True
                            break
                            
                        # 如果是行号匹配
                        if std_start is not None and pred_start is not None:
                            if (std_start <= pred_start <= std_end) or (pred_start <= std_start <= pred_end):
                                found = True
                                break
                    
                    if found:
                        tp += 1
                    else:
                        fn += 1
            else:
                # 文件不匹配
                fn += len(std_file_elements)
        
        # 检查预测中的每个元素是否是假正例
        for pred_file, pred_file_elements in pred_elements.items():
            normalized_pred_file = normalize_path(pred_file)
            
            # 寻找标准中匹配的文件
            matching_std_file = None
            for std_file in std_elements.keys():
                if normalize_path(std_file) == normalized_std_file:
                    matching_std_file = std_file
                    break
            
            if matching_std_file:
                # 找到匹配的文件，检查每个预测元素是否在标准中
                std_file_elements = std_elements[matching_std_file]
                
                for pred_elem in pred_file_elements:
                    pred_type, pred_value, pred_start, pred_end = pred_elem
                    found = False
                    
                    for std_elem in std_file_elements:
                        std_type, std_value, std_start, std_end = std_elem
                        
                        # 匹配类型和值
                        if std_type == pred_type and std_value == pred_value:
                            found = True
                            break
                            
                        # 如果是行号匹配
                        if std_start is not None and pred_start is not None:
                            if (std_start <= pred_start <= std_end) or (pred_start <= std_start <= pred_end):
                                found = True
                                break
                    
                    if not found:
                        fp += 1
            else:
                # 文件不匹配，所有预测元素都是假正例
                fp += len(pred_file_elements)
        
        # 计算当前实例的指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        instance_metrics[instance_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # 处理预测中有但标准中没有的实例
    for instance_id in predictions:
        if instance_id not in standard:
            fp = sum(len(elements) for elements in predictions[instance_id].values())
            total_fp += fp
            instance_metrics[instance_id] = {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": fp, "fn": 0}
    
    # 计算总体指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn
        },
        "instance_metrics": instance_metrics
    }

def compare_models(standard_file, prediction_file1, prediction_file2):
    """比较两个预测模型的性能"""
    # 加载标准数据和预测结果
    standard = load_standard_results(standard_file)
    predictions1 = load_predictions(prediction_file1)
    predictions2 = load_predictions(prediction_file2)
    
    # 评估两个模型
    results1 = evaluate(standard, predictions1)
    results2 = evaluate(standard, predictions2)
    
    # 总体比较
    print(f"{'指标':<15}{'模型1':<15}{'模型2':<15}{'差异':<15}")
    print("="*60)
    for metric in ["precision", "recall", "f1"]:
        value1 = results1["overall"][metric]
        value2 = results2["overall"][metric]
        diff = value2 - value1
        print(f"{metric:<15}{value1:<15.4f}{value2:<15.4f}{diff:<15.4f}")
    
    # 项目级别比较
    print("\n按项目比较 (F1分数):")
    
    # 获取所有项目
    all_instances = set(list(results1["instance_metrics"].keys()) + list(results2["instance_metrics"].keys()))
    projects = defaultdict(list)
    for instance_id in all_instances:
        project = instance_id.split('__')[0]
        projects[project].append(instance_id)
    
    # 计算每个项目的F1分数
    print(f"{'项目':<15}{'模型1':<15}{'模型2':<15}{'差异':<15}")
    print("="*60)
    
    for project, instances in sorted(projects.items()):
        # 提取项目实例的指标
        project_instances1 = [results1["instance_metrics"].get(instance_id, {"f1": 0}) for instance_id in instances]
        project_instances2 = [results2["instance_metrics"].get(instance_id, {"f1": 0}) for instance_id in instances]
        
        # 计算F1
        f1_sum1 = sum(m["f1"] for m in project_instances1)
        f1_sum2 = sum(m["f1"] for m in project_instances2)
        
        # 计算平均F1
        avg_f1_1 = f1_sum1 / len(instances) if instances else 0
        avg_f1_2 = f1_sum2 / len(instances) if instances else 0
        diff = avg_f1_2 - avg_f1_1
        
        print(f"{project:<15}{avg_f1_1:<15.4f}{avg_f1_2:<15.4f}{diff:<15.4f}")
    
    # 比较实例级别性能显著提升的数量
    improved = 0
    regressed = 0
    unchanged = 0
    
    for instance_id in all_instances:
        f1_1 = results1["instance_metrics"].get(instance_id, {"f1": 0})["f1"]
        f1_2 = results2["instance_metrics"].get(instance_id, {"f1": 0})["f1"]
        
        if f1_2 > f1_1:
            improved += 1
        elif f1_2 < f1_1:
            regressed += 1
        else:
            unchanged += 1
    
    print(f"\n实例级别变化:")
    print(f"改进: {improved} ({improved/len(all_instances)*100:.2f}%)")
    print(f"退步: {regressed} ({regressed/len(all_instances)*100:.2f}%)")
    print(f"不变: {unchanged} ({unchanged/len(all_instances)*100:.2f}%)")

def main():
    if len(sys.argv) != 4:
        print("使用方法: python compare.py <标准文件> <预测文件1> <预测文件2>")
        sys.exit(1)
    
    standard_file = sys.argv[1]
    prediction_file1 = sys.argv[2]
    prediction_file2 = sys.argv[3]
    
    compare_models(standard_file, prediction_file1, prediction_file2)

if __name__ == "__main__":
    main() 