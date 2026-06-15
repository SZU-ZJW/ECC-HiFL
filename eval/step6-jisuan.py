import json
import re
import sys
import os
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
                if normalize_path(std_file) == normalized_pred_file:
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
    
    # 修改：使用宏平均计算总体指标，而不是微平均
    total_instances = len(instance_metrics)
    macro_precision = sum(metrics["precision"] for metrics in instance_metrics.values()) / total_instances if total_instances > 0 else 0
    macro_recall = sum(metrics["recall"] for metrics in instance_metrics.values()) / total_instances if total_instances > 0 else 0
    macro_f1 = sum(metrics["f1"] for metrics in instance_metrics.values()) / total_instances if total_instances > 0 else 0
    
    # 同时保留微平均的计算结果
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    return {
        "overall": {
            "precision": macro_precision,  # 使用宏平均
            "recall": macro_recall,        # 使用宏平均
            "f1": macro_f1,                # 使用宏平均
            "micro_precision": micro_precision,  # 保留微平均结果
            "micro_recall": micro_recall,        # 保留微平均结果
            "micro_f1": micro_f1,                # 保留微平均结果
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn
        },
        "instance_metrics": instance_metrics
    }


def analyze_project_performance(instance_metrics):
    """分析各项目的性能"""
    projects = defaultdict(list)
    for instance_id, metrics in instance_metrics.items():
        project = instance_id.split('__')[0]
        projects[project].append(metrics)
    
    project_metrics = {}
    for project, metrics_list in projects.items():
        # 计算平均指标
        avg_precision = sum(m["precision"] for m in metrics_list) / len(metrics_list)
        avg_recall = sum(m["recall"] for m in metrics_list) / len(metrics_list)
        avg_f1 = sum(m["f1"] for m in metrics_list) / len(metrics_list)
        
        # 成功率 (F1 > 0)
        success_count = sum(1 for m in metrics_list if m["f1"] > 0)
        success_rate = success_count / len(metrics_list)
        
        project_metrics[project] = {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "success_rate": success_rate,
            "instance_count": len(metrics_list)
        }
    
    return project_metrics

def analyze_element_types(standard, predictions, instance_metrics):
    """分析不同类型代码元素的性能"""
    element_types = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    # 遍历每个实例
    for instance_id, metrics in instance_metrics.items():
        if instance_id not in standard or instance_id not in predictions:
            continue
        
        std_elements = standard[instance_id]
        pred_elements = predictions[instance_id]
        
        # 统计标准结果中的元素类型
        for file_path, elements in std_elements.items():
            for elem in elements:
                elem_type = elem[0]  # 类型是元组的第一个元素
                
                # 寻找预测中匹配的文件
                matching_pred_file = None
                normalized_file = normalize_path(file_path)
                for pred_file in pred_elements.keys():
                    if normalize_path(pred_file) == normalized_file:
                        matching_pred_file = pred_file
                        break
                
                # 检查元素是否被预测到
                found = False
                if matching_pred_file:
                    pred_file_elements = pred_elements[matching_pred_file]
                    for pred_elem in pred_file_elements:
                        pred_type, pred_value = pred_elem[0], pred_elem[1]
                        
                        if pred_type == elem_type and pred_value == elem[1]:
                            found = True
                            element_types[elem_type]["tp"] += 1
                            break
                
                if not found:
                    element_types[elem_type]["fn"] += 1
        
        # 统计预测结果中的假正例元素类型
        for file_path, elements in pred_elements.items():
            for pred_elem in elements:
                pred_type, pred_value = pred_elem[0], pred_elem[1]
                
                # 寻找标准中匹配的文件
                matching_std_file = None
                normalized_file = normalize_path(file_path)
                for std_file in std_elements.keys():
                    if normalize_path(std_file) == normalized_file:
                        matching_std_file = std_file
                        break
                
                # 检查元素是否在标准中
                found = False
                if matching_std_file:
                    std_file_elements = std_elements[matching_std_file]
                    for std_elem in std_file_elements:
                        std_type, std_value = std_elem[0], std_elem[1]
                        
                        if std_type == pred_type and std_value == pred_value:
                            found = True
                            break
                
                if not found:
                    element_types[pred_type]["fp"] += 1
    
    # 计算每种元素类型的指标
    element_metrics = {}
    for elem_type, counts in element_types.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        element_metrics[elem_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    return element_metrics

def analyze_f1_distribution(instance_metrics):
    """分析F1分数的分布"""
    distribution = {
        "0.0": 0,        # F1 = 0
        "0.0-0.3": 0,    # 0 < F1 <= 0.3
        "0.3-0.5": 0,    # 0.3 < F1 <= 0.5
        "0.5-0.7": 0,    # 0.5 < F1 <= 0.7
        "0.7-0.9": 0,    # 0.7 < F1 <= 0.9
        "0.9-1.0": 0,    # 0.9 < F1 < 1.0
        "1.0": 0         # F1 = 1.0
    }
    
    total_instances = len(instance_metrics)
    for metrics in instance_metrics.values():
        f1 = metrics["f1"]
        
        if f1 == 0:
            distribution["0.0"] += 1
        elif f1 <= 0.3:
            distribution["0.0-0.3"] += 1
        elif f1 <= 0.5:
            distribution["0.3-0.5"] += 1
        elif f1 <= 0.7:
            distribution["0.5-0.7"] += 1
        elif f1 <= 0.9:
            distribution["0.7-0.9"] += 1
        elif f1 < 1.0:
            distribution["0.9-1.0"] += 1
        else:
            distribution["1.0"] += 1
    
    # 计算百分比
    distribution_percent = {
        k: (v / total_instances * 100) for k, v in distribution.items()
    }
    
    return {
        "count": distribution,
        "percent": distribution_percent
    }

def analyze_recall_distribution(instance_metrics):
    """分析召回率分数的分布"""
    distribution = {
        "0.0": 0,        # Recall = 0
        "0.0-0.3": 0,    # 0 < Recall <= 0.3
        "0.3-0.5": 0,    # 0.3 < Recall <= 0.5
        "0.5-0.7": 0,    # 0.5 < Recall <= 0.7
        "0.7-0.9": 0,    # 0.7 < Recall <= 0.9
        "0.9-1.0": 0,    # 0.9 < Recall < 1.0
        "1.0": 0         # Recall = 1.0
    }
    
    total_instances = len(instance_metrics)
    for metrics in instance_metrics.values():
        recall = metrics["recall"]
        
        if recall == 0:
            distribution["0.0"] += 1
        elif recall <= 0.3:
            distribution["0.0-0.3"] += 1
        elif recall <= 0.5:
            distribution["0.3-0.5"] += 1
        elif recall <= 0.7:
            distribution["0.5-0.7"] += 1
        elif recall <= 0.9:
            distribution["0.7-0.9"] += 1
        elif recall < 1.0:
            distribution["0.9-1.0"] += 1
        else:
            distribution["1.0"] += 1
    
    # 计算百分比
    distribution_percent = {
        k: (v / total_instances * 100) for k, v in distribution.items()
    }
    
    return {
        "count": distribution,
        "percent": distribution_percent
    }

def find_low_recall_instances(instance_metrics, threshold=0.5):
    """找出召回率低于阈值的实例，以便进一步分析"""
    low_recall_instances = {}
    for instance_id, metrics in instance_metrics.items():
        if metrics["recall"] < threshold:
            low_recall_instances[instance_id] = metrics
    
    # 按项目分组
    projects_with_low_recall = defaultdict(list)
    for instance_id, metrics in low_recall_instances.items():
        project = instance_id.split('__')[0]
        projects_with_low_recall[project].append((instance_id, metrics))
    
    # 按项目计算低召回率实例的比例
    project_stats = {}
    for project, instances in projects_with_low_recall.items():
        project_stats[project] = {
            "count": len(instances),
            "avg_recall": sum(m["recall"] for _, m in instances) / len(instances) if instances else 0,
            "instances": sorted(instances, key=lambda x: x[1]["recall"])
        }
    
    return {
        "total_count": len(low_recall_instances),
        "by_project": project_stats
    }

def analyze_recall_by_prediction_count(standard, predictions, instance_metrics):
    """分析召回率随预测元素数量变化的趋势"""
    # 按预测元素数量分组统计
    recall_by_count = defaultdict(list)
    
    for instance_id, metrics in instance_metrics.items():
        if instance_id not in predictions:
            continue
        
        # 计算该实例预测的元素总数
        pred_count = sum(len(elements) for elements in predictions[instance_id].values())
        
        # 记录召回率
        recall_by_count[pred_count].append(metrics["recall"])
    
    # 计算每个预测数量的平均召回率
    avg_recall_by_count = {}
    for count, recalls in recall_by_count.items():
        avg_recall_by_count[count] = sum(recalls) / len(recalls)
    
    # 按预测元素数量分组
    count_groups = {
        "1-3": {"recalls": [], "count": 0},
        "4-6": {"recalls": [], "count": 0},
        "7-10": {"recalls": [], "count": 0},
        "11+": {"recalls": [], "count": 0}
    }
    
    for count, recalls in recall_by_count.items():
        if count <= 3:
            count_groups["1-3"]["recalls"].extend(recalls)
            count_groups["1-3"]["count"] += len(recalls)
        elif count <= 6:
            count_groups["4-6"]["recalls"].extend(recalls)
            count_groups["4-6"]["count"] += len(recalls)
        elif count <= 10:
            count_groups["7-10"]["recalls"].extend(recalls)
            count_groups["7-10"]["count"] += len(recalls)
        else:
            count_groups["11+"]["recalls"].extend(recalls)
            count_groups["11+"]["count"] += len(recalls)
    
    # 计算每个组的平均召回率
    group_avg_recall = {}
    for group, data in count_groups.items():
        if data["count"] > 0:
            group_avg_recall[group] = sum(data["recalls"]) / data["count"]
        else:
            group_avg_recall[group] = 0
    
    return {
        "by_exact_count": avg_recall_by_count,
        "by_count_group": group_avg_recall,
        "count_distribution": {group: data["count"] for group, data in count_groups.items()}
    }

def analyze_recall_by_file(standard, predictions):
    """按文件分析召回率"""
    file_recalls = defaultdict(lambda: {"tp": 0, "fn": 0})
    file_instances = defaultdict(int)
    
    # 遍历每个实例
    for instance_id, std_elements in standard.items():
        if instance_id not in predictions:
            # 如果预测结果中没有该实例，增加所有文件的FN
            for file_path, elements in std_elements.items():
                normalized_file = normalize_path(file_path)
                file_recalls[normalized_file]["fn"] += len(elements)
                file_instances[normalized_file] += 1
            continue
        
        pred_elements = predictions[instance_id]
        
        # 对标准结果中的每个文件
        for std_file, std_file_elements in std_elements.items():
            normalized_std_file = normalize_path(std_file)
            file_instances[normalized_std_file] += 1
            
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
                        file_recalls[normalized_std_file]["tp"] += 1
                    else:
                        file_recalls[normalized_std_file]["fn"] += 1
            else:
                # 文件不匹配，所有标准元素都是FN
                file_recalls[normalized_std_file]["fn"] += len(std_file_elements)
    
    # 计算每个文件的召回率
    file_metrics = {}
    for file_path, counts in file_recalls.items():
        tp = counts["tp"]
        fn = counts["fn"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        instances = file_instances[file_path]
        
        file_metrics[file_path] = {
            "recall": recall,
            "tp": tp,
            "fn": fn,
            "total_elements": tp + fn,
            "instance_count": instances
        }
    
    return file_metrics

def main():
    standard_file = "data/legacy/GRM4-V/others/SWE-Bench-Verified-element-level-line_number.jsonl"
    standard = load_standard_results(standard_file)
    sample_num_list = [1,3,7,10,15,30]
    for sample_num in sample_num_list:
        print(f"============= {sample_num} =============")
        max = 0
        index = 0
        result = None  # 改为None，更明确地表示未初始化状态
        sum_recall = 0
        for i in range(1, 2):
            
            line_loc_path = f"results/sk-{sample_num}/1/edit_location_samples/loc_outputs.jsonl"
            prediction_file = "sample.json"
            if os.path.exists(line_loc_path):
                print(f"------- {i} ---------")
                with open(prediction_file, 'w') as savef:
                    with open(line_loc_path, 'r') as locf:
                        for line in locf:
                            data = json.loads(line)
                            instance_id = data["instance_id"]
                            edit_loc = data["found_edit_locs"]
                            metadata = {
                                "instance_id": instance_id,
                                "edit_loc": edit_loc
                            }
                            json.dump(metadata, savef)
                            savef.write("\n")
                            

                predictions = load_predictions(prediction_file)
                results = evaluate(standard, predictions)

                overall = results["overall"]
                # 修改逻辑：确保result总是被赋值
                if result is None or max < overall['recall']:
                    max = overall['recall']
                    result = overall
                    index = i
                precision = overall['precision']
                recall = overall['recall']
                f1 = overall['f1']

                sum_recall += recall
                print(f" Prec : {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"  F1  : {f1:.4f}\n")
            else:
                print(f"文件不存在: {line_loc_path}")

        # 添加安全检查
        if result is not None:
            # print(f"sum_recall: {sum_recall/20:.4f}")
            # print(f" Prec : {result['precision']:.4f}")
            print(f"Recall: {result['recall']:.4f}")
            # print(f"  F1  : {result['f1']:.4f}")
            # print(f"index: {index}")
        else:
            print("No valid results found for this sample_num")
if __name__ == "__main__":
    main() 