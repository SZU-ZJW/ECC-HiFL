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
    if len(sys.argv) != 3:
        print("使用方法: python evaluate.py <标准文件> <预测文件>")
        sys.exit(1)
    
    standard_file = sys.argv[1]
    prediction_file = sys.argv[2]
    
    # 加载标准数据和预测结果
  #  print(f"加载标准结果: {standard_file}")
    standard = load_standard_results(standard_file)
   # print(f"加载预测结果: {prediction_file}")
    predictions = load_predictions(prediction_file)
    
    # 评估预测结果
  #  print("评估预测结果...")
    results = evaluate(standard, predictions)
    
    # 打印总体评估指标
    overall = results["overall"]
   # print("\n=== 总体评估指标 ===")
    print(f" Prec : {overall['precision']:.4f}")
    print(f"Recall: {overall['recall']:.4f}")
    print(f"  F1  : {overall['f1']:.4f}")
    # print(f"真正例 (TP): {overall['tp']}")
    # print(f"假正例 (FP): {overall['fp']}")
    # print(f"假反例 (FN): {overall['fn']}")
    
    # # 分析项目性能, 重点关注Recall
    # print("\n=== 项目性能分析 (按Recall排序) ===")
    # project_metrics = analyze_project_performance(results["instance_metrics"])
    # print(f"{'项目':<15}{'召回率':<10}{'精确率':<10}{'平均F1':<10}{'成功率':<10}{'实例数':<10}")
    # print("-" * 65)
    # for project, metrics in sorted(project_metrics.items(), key=lambda x: x[1]["avg_recall"], reverse=True):
    #     print(f"{project:<15}{metrics['avg_recall']:<10.4f}{metrics['avg_precision']:<10.4f}{metrics['avg_f1']:<10.4f}{metrics['success_rate']:<10.2f}{metrics['instance_count']:<10}")
    
    # # 分析F1分布
    # print("\n=== F1分数分布 ===")
    # f1_distribution = analyze_f1_distribution(results["instance_metrics"])
    # print(f"{'F1分数区间':<15}{'实例数':<10}{'百分比':<10}")
    # print("-" * 35)
    # for range_name, count in f1_distribution["count"].items():
    #     percent = f1_distribution["percent"][range_name]
    #     print(f"{range_name:<15}{count:<10}{percent:.2f}%")
    
    # # 分析Recall分布
    # print("\n=== 召回率分布 ===")
    # recall_distribution = analyze_recall_distribution(results["instance_metrics"])
    # print(f"{'召回率区间':<15}{'实例数':<10}{'百分比':<10}")
    # print("-" * 35)
    # for range_name, count in recall_distribution["count"].items():
    #     percent = recall_distribution["percent"][range_name]
    #     print(f"{range_name:<15}{count:<10}{percent:.2f}%")
    
    # # 分析低召回率实例
    # print("\n=== 低召回率实例分析 (Recall < 0.5) ===")
    # low_recall_analysis = find_low_recall_instances(results["instance_metrics"], 0.5)
    # print(f"总共有 {low_recall_analysis['total_count']} 个实例召回率低于0.5")
    # print("\n按项目统计低召回率实例：")
    # print(f"{'项目':<15}{'低召回率实例数':<15}{'平均召回率':<15}")
    # print("-" * 45)
    # for project, stats in sorted(low_recall_analysis['by_project'].items(), key=lambda x: x[1]["count"], reverse=True):
    #     print(f"{project:<15}{stats['count']:<15}{stats['avg_recall']:<15.4f}")
    
    # # 输出每个项目中召回率最低的三个实例
    # print("\n每个项目中召回率最低的3个实例：")
    # for project, stats in sorted(low_recall_analysis['by_project'].items(), key=lambda x: x[1]["avg_recall"]):
    #     print(f"\n项目: {project} (平均召回率: {stats['avg_recall']:.4f})")
        # for i, (instance_id, metrics) in enumerate(stats['instances'][:3]):
        #     print(f"  {i+1}. {instance_id}: 召回率={metrics['recall']:.4f}, 精确率={metrics['precision']:.4f}, F1={metrics['f1']:.4f}")
    
    # 分析召回率随预测元素数量的变化
    # print("\n=== 召回率与预测元素数量的关系 ===")
    # count_analysis = analyze_recall_by_prediction_count(standard, predictions, results["instance_metrics"])
    
    # print("\n按预测元素数量组统计：")
    # print(f"{'元素数量组':<10}{'实例数':<10}{'平均召回率':<15}")
    # print("-" * 35)
    # for group, count in sorted(count_analysis["count_distribution"].items(), key=lambda x: x[0]):
    #     avg_recall = count_analysis["by_count_group"][group]
    #     print(f"{group:<10}{count:<10}{avg_recall:.4f}")
    
    # # 分析按文件的召回率
    # print("\n=== 按文件类型分析召回率 ===")
    # file_analysis = analyze_recall_by_file(standard, predictions)
    
    # # 只显示出现在至少3个实例中的文件
    # min_instances = 3
    # relevant_files = {f: m for f, m in file_analysis.items() if m["instance_count"] >= min_instances}
    
    # print(f"\n至少在{min_instances}个实例中出现的文件 (按召回率排序)：")
    # print(f"{'文件类型':<25}{'召回率':<10}{'元素总数':<10}{'实例数':<10}")
    # print("-" * 55)
    # for file_path, metrics in sorted(relevant_files.items(), key=lambda x: x[1]["recall"], reverse=True):
    #     print(f"{file_path:<25}{metrics['recall']:<10.4f}{metrics['total_elements']:<10}{metrics['instance_count']:<10}")
    
    # # 打印召回率最低的10个文件
    # print(f"\n召回率最低的文件 (至少在{min_instances}个实例中出现)：")
    # low_recall_files = sorted(relevant_files.items(), key=lambda x: x[1]["recall"])[:10]
    # for file_path, metrics in low_recall_files:
    #     print(f"{file_path:<25}{metrics['recall']:<10.4f}{metrics['total_elements']:<10}{metrics['instance_count']:<10}")
    
    # 分析元素类型，重点关注Recall
    # print("\n=== 代码元素类型分析 (按Recall排序) ===")
    # element_metrics = analyze_element_types(standard, predictions, results["instance_metrics"])
    # print(f"{'元素类型':<15}{'召回率':<10}{'精确率':<10}{'F1分数':<10}{'TP':<8}{'FP':<8}{'FN':<8}")
    # print("-" * 69)
    # for elem_type, metrics in sorted(element_metrics.items(), key=lambda x: x[1]["recall"], reverse=True):
    #     print(f"{elem_type:<15}{metrics['recall']:<10.4f}{metrics['precision']:<10.4f}{metrics['f1']:<10.4f}{metrics['tp']:<8}{metrics['fp']:<8}{metrics['fn']:<8}")
    
    # # 打印实例级评估结果，重点关注Recall
    # print("\n=== 实例级评估结果 (按Recall排序) ===")
    # print(f"{'实例ID':<30}{'召回率':<10}{'精确率':<10}{'F1分数':<10}")
    # print("-" * 60)
    # for instance_id, metrics in sorted(results["instance_metrics"].items(), key=lambda x: x[1]["recall"], reverse=True):
    #     print(f"{instance_id:<30}{metrics['recall']:<10.4f}{metrics['precision']:<10.4f}{metrics['f1']:<10.4f}")

if __name__ == "__main__":
    main() 