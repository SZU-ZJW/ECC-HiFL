"""Localization metrics, ported faithfully from ``eval/step1-jisuan.py`` (file),
``eval/step5-jisuan.py`` (function/element) and ``eval/step6-jisuan.py`` (line).

Each metric function consumes ``preds`` = ``{instance_id: prediction_row}`` (rows in the
upstream ``loc_outputs.jsonl`` shape — ``found_files`` / ``found_related_locs`` /
``found_edit_locs``) and a gold map, and returns an aggregate dict plus per-instance details.
By default we aggregate over ``set(gold) & set(preds)`` (the shared instance set every selector
predicts on), reporting ``n``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..io_utils import load_jsonl


def _normalize_path(path: str) -> str:
    """Match files by basename (matches step5/step6 ``normalize_path``)."""
    return path.split("/")[-1]


# ============================================================ file level (step1)

def load_file_gold(path: str) -> Dict[str, List[str]]:
    """Gold modified-files map, auto-detecting the format.

    Accepts either ``modified_files.jsonl`` rows (``{instance_id, modified_files:[...]}``) or
    element-level rows (``{instance_id, result:{code_elements:{file:[...]}}}``) — in the latter
    case the gold file set is the set of ``code_elements`` keys. This lets Verified (which ships
    only element-level gold) be evaluated at file level too.
    """
    out = {}
    for row in load_jsonl(path):
        if "modified_files" in row:
            out[row["instance_id"]] = row.get("modified_files", [])
        else:
            ce = (row.get("result") or {}).get("code_elements", {})
            out[row["instance_id"]] = list(ce.keys())
    return out


def file_metrics(preds: Dict[str, dict], gold: Dict[str, List[str]]) -> dict:
    eval_ids = [i for i in gold if i in preds and gold[i]]
    agg = defaultdict(float)
    em = top1 = top3 = top5 = 0
    per_instance = {}
    for iid in eval_ids:
        modified = set(gold[iid])
        found = preds[iid].get("found_files", []) or []
        pred_set = set(found)
        tp = len(modified & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(modified) if modified else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        is_em = modified == pred_set
        hit1 = bool(modified & set(found[:1]))
        hit3 = bool(modified & set(found[:3]))
        hit5 = bool(modified & set(found[:5]))
        agg["precision"] += precision
        agg["recall"] += recall
        agg["f1"] += f1
        em += int(is_em)
        top1 += int(hit1)
        top3 += int(hit3)
        top5 += int(hit5)
        per_instance[iid] = {"precision": precision, "recall": recall, "f1": f1,
                             "em": is_em, "top1": hit1, "top3": hit3, "top5": hit5}
    n = len(eval_ids)
    return {
        "level": "file", "n": n,
        "precision": agg["precision"] / n if n else 0.0,
        "recall": agg["recall"] / n if n else 0.0,
        "f1": agg["f1"] / n if n else 0.0,
        "em": em / n if n else 0.0,
        "top1": top1 / n if n else 0.0,
        "top3": top3 / n if n else 0.0,
        "top5": top5 / n if n else 0.0,
        "per_instance": per_instance,
    }


# ======================================================= function level (step5)

def load_element_gold(path: str) -> Dict[str, set]:
    """element-level gold -> {instance_id: {"file:type: value", ...}} (step5 ``extract_code_elements``)."""
    out = {}
    for row in load_jsonl(path):
        elements = set()
        ce = (row.get("result") or {}).get("code_elements", {})
        for file_path, items in ce.items():
            for item in items:
                if isinstance(item, dict) and "type" in item and "value" in item:
                    elements.add(f"{file_path}:{item['type']}: {item['value']}")
                else:
                    elements.add(f"{file_path}:{item}")
        out[row["instance_id"]] = elements
    return out


def _predicted_elements(pred_row: dict) -> set:
    """found_related_locs -> {"file:function: x", ...} (step5 ``extract_predicted_elements``)."""
    elements = set()
    for file_path, items in (pred_row.get("found_related_locs") or {}).items():
        for item in items:
            if not item:
                continue
            for part in item.split("\n"):
                if part:
                    elements.add(f"{file_path}:{part}")
    return elements


def _is_class_method_match(gold_element: str, pred_element: str) -> bool:
    """Class<->method relationship match (verbatim from step5 ``is_class_method_match``)."""
    gold_parts = gold_element.split(":", 1)
    pred_parts = pred_element.split(":", 1)
    if len(gold_parts) < 2 or len(pred_parts) < 2:
        return False
    if gold_parts[0] != pred_parts[0]:  # file must match
        return False
    gold_content = gold_parts[1].strip()
    pred_content = pred_parts[1].strip()
    gold_tv = gold_content.split(":", 1) if ":" in gold_content else ["", gold_content]
    pred_tv = pred_content.split(":", 1) if ":" in pred_content else ["", pred_content]
    gold_value = gold_tv[1].strip() if len(gold_tv) > 1 else gold_content
    pred_value = pred_tv[1].strip() if len(pred_tv) > 1 else pred_content
    if ("class:" in gold_content and "function:" in pred_content) or (
        "function:" in gold_content and "class:" in pred_content
    ):
        gold_class = gold_value.split(".")[0] if "." in gold_value else gold_value
        pred_class = pred_value.split(".")[0] if "." in pred_value else pred_value
        return (gold_class == pred_class or gold_value == pred_class
                or pred_value == gold_class or gold_value in pred_value or pred_value in gold_value)
    return False


def function_metrics(preds: Dict[str, dict], gold: Dict[str, set]) -> dict:
    eval_ids = sorted(set(gold) & set(preds))
    total_tp = total_fp = total_fn = 0
    per_instance = {}
    for iid in eval_ids:
        gold_elems = gold[iid]
        pred_elems = _predicted_elements(preds[iid])
        matched_gold, matched_pred = set(), set()
        tp = 0
        for ge in gold_elems:  # exact first
            for pe in pred_elems:
                if ge == pe and pe not in matched_pred:
                    tp += 1; matched_gold.add(ge); matched_pred.add(pe); break
        for ge in gold_elems - matched_gold:  # then class<->method
            for pe in pred_elems - matched_pred:
                if _is_class_method_match(ge, pe):
                    tp += 1; matched_gold.add(ge); matched_pred.add(pe); break
        fp = len(pred_elems - matched_pred)
        fn = len(gold_elems - matched_gold)
        total_tp += tp; total_fp += fp; total_fn += fn
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_instance[iid] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    n = len(eval_ids)
    macro_p = sum(m["precision"] for m in per_instance.values()) / n if n else 0.0
    macro_r = sum(m["recall"] for m in per_instance.values()) / n if n else 0.0
    macro_f1 = sum(m["f1"] for m in per_instance.values()) / n if n else 0.0
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return {
        "level": "function", "n": n,
        "precision": macro_p, "recall": macro_r, "f1": macro_f1,
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "per_instance": per_instance,
    }


# =========================================================== line level (step6)

ElementTuple = Tuple[str, str, Optional[int], Optional[int]]


def load_line_gold(path: str) -> Dict[str, Dict[str, List[ElementTuple]]]:
    """line-number gold -> {iid: {file: [(type,value,start,end)]}} (step6 ``load_standard_results``)."""
    out = {}
    for row in load_jsonl(path):
        ce = {}
        for file_path, elements in (row.get("result") or {}).get("code_elements", {}).items():
            ce[file_path] = []
            for el in elements:
                if "start_line" in el and "end_line" in el:
                    ce[file_path].append((el["type"], el["value"], el["start_line"], el["end_line"]))
                else:
                    ce[file_path].append((el["type"], el["value"], None, None))
        out[row["instance_id"]] = ce
    return out


def _predicted_line_elements(pred_row: dict) -> Dict[str, List[ElementTuple]]:
    """found_edit_locs -> {file: [(type,value,start,end)]} (step6 ``load_predictions``)."""
    out: Dict[str, List[ElementTuple]] = defaultdict(list)
    for file_path, elements in (pred_row.get("found_edit_locs") or {}).items():
        if not elements or (len(elements) == 1 and elements[0] == ""):
            continue
        for element in elements:
            if not element:
                continue
            func_match = re.search(r"function:\s*(\w+(?:\.\w+)*)", element)
            class_match = re.search(r"class:\s*(\w+(?:\.\w+)*)", element)
            line_match = re.search(r"line:\s*(\d+)", element)
            if func_match:
                out[file_path].append(("function", func_match.group(1), None, None))
            if class_match:
                out[file_path].append(("class", class_match.group(1), None, None))
            if line_match and not func_match and not class_match:
                ln = int(line_match.group(1))
                out[file_path].append(("line", str(ln), ln, ln))
    return dict(out)


def _line_elem_match(std: ElementTuple, pred: ElementTuple) -> bool:
    st_type, st_val, st_s, st_e = std
    pr_type, pr_val, pr_s, pr_e = pred
    if st_type == pr_type and st_val == pr_val:
        return True
    if st_s is not None and pr_s is not None:
        if (st_s <= pr_s <= st_e) or (pr_s <= st_s <= pr_e):
            return True
    return False


def line_metrics(preds: Dict[str, dict], gold: Dict[str, Dict[str, List[ElementTuple]]]) -> dict:
    eval_ids = sorted(set(gold) & set(preds))
    total_tp = total_fp = total_fn = 0
    per_instance = {}
    for iid in eval_ids:
        std_elements = gold[iid]
        pred_elements = _predicted_line_elements(preds[iid])
        tp = fp = fn = 0
        # recall side: every gold element
        for std_file, std_items in std_elements.items():
            nfile = _normalize_path(std_file)
            match_pred_file = next((pf for pf in pred_elements if _normalize_path(pf) == nfile), None)
            if match_pred_file:
                for se in std_items:
                    if any(_line_elem_match(se, pe) for pe in pred_elements[match_pred_file]):
                        tp += 1
                    else:
                        fn += 1
            else:
                fn += len(std_items)
        # precision side: every pred element
        for pred_file, pred_items in pred_elements.items():
            nfile = _normalize_path(pred_file)
            match_std_file = next((sf for sf in std_elements if _normalize_path(sf) == nfile), None)
            if match_std_file:
                for pe in pred_items:
                    if not any(_line_elem_match(se, pe) for se in std_elements[match_std_file]):
                        fp += 1
            else:
                fp += len(pred_items)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_instance[iid] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        total_tp += tp; total_fp += fp; total_fn += fn
    n = len(eval_ids)
    macro_p = sum(m["precision"] for m in per_instance.values()) / n if n else 0.0
    macro_r = sum(m["recall"] for m in per_instance.values()) / n if n else 0.0
    macro_f1 = sum(m["f1"] for m in per_instance.values()) / n if n else 0.0
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return {
        "level": "line", "n": n,
        "precision": macro_p, "recall": macro_r, "f1": macro_f1,
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "per_instance": per_instance,
    }


# ----------------------------------------------------------------- dispatch

def load_gold(level: str, path: str):
    return {"file": load_file_gold, "function": load_element_gold, "line": load_line_gold}[level](path)


def compute_metrics(level: str, preds: Dict[str, dict], gold) -> dict:
    return {"file": file_metrics, "function": function_metrics, "line": line_metrics}[level](preds, gold)


def format_metrics(m: dict) -> str:
    lvl = m["level"]
    if lvl == "file":
        return (f"[file] n={m['n']}  P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                f"EM={m['em']:.4f}  Top1={m['top1']:.4f} Top3={m['top3']:.4f} Top5={m['top5']:.4f}")
    return (f"[{lvl}] n={m['n']}  macro P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} | "
            f"micro P={m['micro_precision']:.4f} R={m['micro_recall']:.4f} F1={m['micro_f1']:.4f}")
