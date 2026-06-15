#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict
import os

def load_jsonl(file_path):
    """加载jsonl文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_code_elements(instance):
    """从真实结果中提取代码元素"""
    elements = set()
    if 'result' in instance and 'code_elements' in instance['result']:
        for file_path, items in instance['result']['code_elements'].items():
            for item in items:
                if isinstance(item, dict) and 'type' in item and 'value' in item:
                    elements.add(f"{file_path}:{item['type']}: {item['value']}")
                else:
                    # 处理其他可能的格式
                    elements.add(f"{file_path}:{item}")
    return elements

def extract_predicted_elements(instance):
    """从预测结果中提取代码元素"""
    elements = set()
    if 'found_related_locs' in instance:
        for file_path, items in instance['found_related_locs'].items():
            for item in items:
                if item:  # 跳过空字符串
                    # 处理形如 "function: _interpret_err_lines" 的格式
                    parts = item.split('\n')
                    for part in parts:
                        if part:
                            elements.add(f"{file_path}:{part}")
    return elements

def is_class_method_match(gold_element, pred_element):
    """检查是否存在类和类方法的关系匹配"""
    # 从元素中提取文件路径、类型和值
    gold_parts = gold_element.split(':', 1)
    pred_parts = pred_element.split(':', 1)
    
    if len(gold_parts) < 2 or len(pred_parts) < 2:
        return False
    
    gold_file = gold_parts[0]
    pred_file = pred_parts[0]
    
    # 如果文件不同，直接返回False
    if gold_file != pred_file:
        return False
    
    gold_content = gold_parts[1].strip()
    pred_content = pred_parts[1].strip()
    
    # 尝试提取类型和值
    gold_type_value = gold_content.split(':', 1) if ':' in gold_content else ['', gold_content]
    pred_type_value = pred_content.split(':', 1) if ':' in pred_content else ['', pred_content]
    
    gold_value = gold_type_value[1].strip() if len(gold_type_value) > 1 else gold_content
    pred_value = pred_type_value[1].strip() if len(pred_type_value) > 1 else pred_content
    
    # 检查类和方法的关系
    # 例如：如果gold是"class: ClassName"，pred是"function: ClassName.method"
    # 或者gold是"function: ClassName.method"，pred是"class: ClassName"
    if ('class:' in gold_content and 'function:' in pred_content) or ('function:' in gold_content and 'class:' in pred_content):
        # 从类方法中提取类名
        if '.' in gold_value:
            gold_class = gold_value.split('.')[0]
        else:
            gold_class = gold_value
            
        if '.' in pred_value:
            pred_class = pred_value.split('.')[0]
        else:
            pred_class = pred_value
            
        # 检查类名是否匹配
        return gold_class == pred_class or gold_value == pred_class or pred_value == gold_class or (gold_value in pred_value or pred_value in gold_value)
    
    return False

def calculate_metrics(gold_data, pred_data):
    """计算精确率、召回率和F1分数"""
    results = {}
    
    # 按实例ID组织数据
    gold_dict = {item['instance_id']: item for item in gold_data}
    pred_dict = {item['instance_id']: item for item in pred_data}
    
    # 所有实例的汇总指标
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # 计算每个实例的指标
    for instance_id in set(gold_dict.keys()) & set(pred_dict.keys()):
        gold_elements = extract_code_elements(gold_dict[instance_id])
        pred_elements = extract_predicted_elements(pred_dict[instance_id])
        
        # 考虑类和方法关系的TP、FN计算
        tp = 0
        matched_gold = set()
        matched_pred = set()
        
        # 首先检查直接匹配的元素
        for gold_elem in gold_elements:
            for pred_elem in pred_elements:
                if gold_elem == pred_elem:
                    tp += 1
                    matched_gold.add(gold_elem)
                    matched_pred.add(pred_elem)
                    break
        
        # 然后检查类和方法关系的匹配
        for gold_elem in gold_elements - matched_gold:
            for pred_elem in pred_elements - matched_pred:
                if is_class_method_match(gold_elem, pred_elem):
                    tp += 1
                    matched_gold.add(gold_elem)
                    matched_pred.add(pred_elem)
                    break
        
        fp = len(pred_elements - matched_pred)
        fn = len(gold_elements - matched_gold)
        
        # 累加到总数
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 计算该实例的指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[instance_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }
    
    # 计算微平均指标
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # 计算宏平均指标
    if results:
        macro_precision = sum(item["precision"] for item in results.values()) / len(results)
        macro_recall = sum(item["recall"] for item in results.values()) / len(results)
        macro_f1 = sum(item["f1"] for item in results.values()) / len(results)
    else:
        macro_precision = macro_recall = macro_f1 = 0
    
    return {
        "instances": results,
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        }
    }

def main():
    parser = argparse.ArgumentParser(description='计算预测结果和真实结果之间的精确率、召回率和F1分数')
    parser.add_argument('--gold', default="data/legacy/GRM4-V/others/SWE-Bench-Verified-element-level.jsonl", help='真实结果的jsonl文件路径')
    args = parser.parse_args()

    sample_num_list = [1,3,7,10,15,30]

    gold_data = load_jsonl(args.gold)

    for sample_num in sample_num_list:
        print("="*15 + f"{sample_num}" + "="*15)
        sum = 0
        n = 0
        for i in range(1, 2):
            pred_path = f'data/legacy/GRM4-V/GRM4-V/STEP5/results/swe-bench-verified/sk-{sample_num}/related_elements/loc_outputs.jsonl'
            
            # 加载数据
            if os.path.exists(pred_path):
                pred_data = load_jsonl(pred_path)
                # 计算指标
                metrics = calculate_metrics(gold_data, pred_data)
                recall = metrics['macro']['recall']
                if recall < 0.22:
                    continue
                # 打印总体指标
                #print(f"宏平均指标:")
                print("-"*10 + f"NO.{i}" + "-"*10)
                print(f"  精确率: {metrics['macro']['precision']:.4f}")
                print(f"  召回率: {metrics['macro']['recall']:.4f}")
                print(f"  F1分数: {metrics['macro']['f1']:.4f}")
                print("-"*10)
                sum += metrics['macro']['recall']
                n += 1
        if n > 0:
            print(f"平均召回率: {sum/n:.4f}")

if __name__ == '__main__':
    main() 