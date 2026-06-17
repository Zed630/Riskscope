import argparse
import json
import os
import re
from collections import Counter, defaultdict


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse line {line_no} in {path}: {e}")
    return items


def get_image_paths(item):
    if not isinstance(item.get("images"), list):
        return []
    paths = []
    for img in item["images"]:
        if isinstance(img, dict):
            p = img.get("path", "")
        else:
            p = img
        if p:
            paths.append(p)
    return paths


def get_source_from_image(image_path):
    if not image_path:
        return "UNKNOWN"
    parts = [p for p in image_path.strip("/").split("/") if p]
    if len(parts) >= 3:
        return parts[2]
    return "UNKNOWN"


def get_user_content(item):
    for msg in item.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def build_key(item):
    img_paths = tuple(get_image_paths(item))
    user_content = get_user_content(item)
    return (img_paths, user_content)


def parse_solution(solution):
    if not solution:
        return None, []
    solution = solution.strip()
    if "," in solution:
        label_part, cat_part = solution.split(",", 1)
    else:
        label_part, cat_part = solution, "[]"
    label_part = label_part.strip()
    is_harmful = label_part.lower() == "harmful"
    cats = []
    try:
        cats = json.loads(cat_part.strip())
        if not isinstance(cats, list):
            cats = []
    except Exception:
        m = re.findall(r"\[([^\]]*)\]", cat_part)
        if m:
            inside = m[0]
            cats = [c.strip() for c in inside.split(",") if c.strip()]
    return is_harmful, cats


def parse_response(response_text):
    if not response_text:
        return None, []
    text = response_text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            is_h = data.get("is_harmful", None)
            cats = data.get("risk_categories", []) or []
            return is_h, cats
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                is_h = data.get("is_harmful", None)
                cats = data.get("risk_categories", []) or []
                return is_h, cats
        except Exception:
            pass
    m = re.search(r'"is_harmful"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true", []
    return None, []


def compute_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = (tp + tn) / total * 100 if total else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 0
    pre = tp / (tp + fp) * 100 if (tp + fp) else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
    return acc, rec, pre, f1


def merge_with_label_file(dataset, label_file):
    """从 label_file 中加载 solution 字段，按 (图片路径 + 用户消息) 匹配合并到 dataset"""
    if not label_file:
        return dataset, 0, 0

    label_items = load_jsonl(label_file)
    label_map = {}
    for item in label_items:
        sol = item.get("solution")
        if sol:
            key = build_key(item)
            label_map[key] = sol

    merged = []
    matched_count = 0
    for item in dataset:
        key = build_key(item)
        if key in label_map:
            new_item = dict(item)
            new_item["solution"] = label_map[key]
            merged.append(new_item)
            matched_count += 1
        else:
            merged.append(item)

    print(f"    使用 label 文件: {label_file} ({len(label_items)} 条)")
    print(f"    成功匹配 solution: {matched_count} / {len(dataset)}")
    return merged, matched_count, len(label_items)


def analyze_source_distribution(dataset):
    source_counter = Counter()
    per_source_label = defaultdict(lambda: {"harmful": 0, "safe": 0, "total": 0})
    for item in dataset:
        img_paths = get_image_paths(item)
        is_harmful, _ = parse_solution(item.get("solution"))
        if img_paths:
            src = get_source_from_image(img_paths[0])
        else:
            src = "UNKNOWN"
        source_counter[src] += 1
        per_source_label[src]["total"] += 1
        if is_harmful is True:
            per_source_label[src]["harmful"] += 1
        elif is_harmful is False:
            per_source_label[src]["safe"] += 1
    return source_counter, per_source_label


def analyze_label_distribution(dataset):
    label_counter = Counter()
    category_counter = Counter()
    per_cat_harmful = Counter()
    for item in dataset:
        is_harmful, cats = parse_solution(item.get("solution"))
        if is_harmful is True:
            label_counter["Harmful"] += 1
        elif is_harmful is False:
            label_counter["Safe"] += 1
        else:
            label_counter["Unknown"] += 1
        for c in cats:
            category_counter[c] += 1
            if is_harmful is True:
                per_cat_harmful[c] += 1
    return label_counter, category_counter, per_cat_harmful


def evaluate_inference(dataset):
    all_valid_items = []
    all_categories = set()
    parse_fail = 0
    for item in dataset:
        gt_harmful, gt_cats = parse_solution(item.get("solution"))
        if gt_harmful is None:
            continue
        pred_harmful, _ = parse_response(item.get("response"))
        if pred_harmful is None:
            parse_fail += 1
            pred_harmful = False
        gt_cats_set = set(gt_cats)
        all_categories.update(gt_cats_set)
        all_valid_items.append((gt_harmful, gt_cats_set, pred_harmful))

    all_categories = sorted(all_categories)
    matched = len(all_valid_items)

    tp = fp = fn = tn = 0
    for gt_h, gt_c, pred_h in all_valid_items:
        if gt_h and pred_h:
            tp += 1
        elif not gt_h and pred_h:
            fp += 1
        elif gt_h and not pred_h:
            fn += 1
        else:
            tn += 1

    acc, rec, pre, f1 = compute_metrics(tp, fp, fn, tn)
    binary_m = {"total": tp + fp + fn + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "acc": acc, "recall": rec, "precision": pre, "f1": f1}

    per_cat_m = {}
    for c in all_categories:
        ctp = cfp = cfn = ctn = 0
        for gt_h, gt_c, pred_h in all_valid_items:
            if c not in gt_c:
                continue
            if gt_h and pred_h:
                ctp += 1
            elif not gt_h and pred_h:
                cfp += 1
            elif gt_h and not pred_h:
                cfn += 1
            else:
                ctn += 1
        a, r, p, f = compute_metrics(ctp, cfp, cfn, ctn)
        per_cat_m[c] = {"total": ctp + cfp + cfn + ctn, "tp": ctp, "fp": cfp, "fn": cfn, "tn": ctn,
                        "acc": a, "recall": r, "precision": p, "f1": f}

    return matched, parse_fail, binary_m, per_cat_m, all_categories


def main():
    parser = argparse.ArgumentParser(description="数据集统计 & 推理结果评估")
    parser.add_argument("input_file", help="输入 jsonl 文件路径，如推理结果文件 (包含 response 和 images)")
    parser.add_argument("output_file", help="输出 JSON 报告路径")
    parser.add_argument("--label-file", dest="label_file", default=None,
                        help="可选: 包含 solution 标签的 jsonl 文件。当 input_file 中没有 solution 时用于匹配。")
    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file
    label_path = args.label_file

    print("=" * 70)
    print("数据集分布 & 推理结果分析")
    print("=" * 70)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    if label_path:
        print(f"标签文件: {label_path}")

    if not os.path.exists(input_path):
        print(f"[ERROR] 文件不存在: {input_path}")
        return

    if label_path and not os.path.exists(label_path):
        print(f"[ERROR] 标签文件不存在: {label_path}")
        return

    dataset = load_jsonl(input_path)
    dataset_size = len(dataset)
    print(f"\n[1] 数据集总条目数: {dataset_size}")

    # 合并标签文件 (如果提供)
    if label_path:
        dataset, label_matched, label_total = merge_with_label_file(dataset, label_path)

    print(f"\n[2] 数据来源构成 (图片路径第三级目录):")
    source_counter, per_source_label = analyze_source_distribution(dataset)
    total_items = sum(source_counter.values())
    print(f"    共 {len(source_counter)} 个来源，条目总数: {total_items}")
    print(f"    {'来源':<40} {'数量':>8} {'占比':>8} {'有害':>8} {'安全':>8}")
    print("    " + "-" * 80)
    for src, cnt in source_counter.most_common():
        ratio = (cnt / dataset_size * 100) if dataset_size else 0
        lbl = per_source_label[src]
        print(f"    {src:<40} {cnt:>8} {ratio:>7.2f}% {lbl['harmful']:>8} {lbl['safe']:>8}")

    print(f"\n[3] 正负样本数量占比:")
    label_counter, category_counter, per_cat_harmful = analyze_label_distribution(dataset)
    print(f"    {'标签':<15} {'数量':>10} {'占比':>10}")
    print("    " + "-" * 40)
    for lbl, cnt in label_counter.most_common():
        print(f"    {lbl:<15} {cnt:>10} {cnt/dataset_size*100:>9.2f}%")

    print(f"\n[4] 各类别样本数量及有害数目占比:")
    print(f"    {'类别':<50} {'数量':>10} {'占比':>10} {'有害':>8} {'有害占比':>10}")
    print("    " + "-" * 100)
    for cat, cnt in category_counter.most_common():
        h = per_cat_harmful.get(cat, 0)
        hr = (h / cnt * 100) if cnt else 0
        print(f"    {cat:<50} {cnt:>10} {cnt/dataset_size*100:>9.2f}% {h:>8} {hr:>9.2f}%")

    print(f"\n[5] 模型推理指标 (ACC / REC / PRE / F1):")
    matched, parse_fail, binary_m, per_cat_m, all_categories = evaluate_inference(dataset)
    print(f"    有效条目(solution可解析): {matched}")
    print(f"    response解析失败(按无害处理): {parse_fail}")

    b = binary_m
    print(f"\n    [整体二分类] is_harmful 指标:")
    print(f"      总样本: {b['total']}, TP={b['tp']}, FP={b['fp']}, FN={b['fn']}, TN={b['tn']}")
    print(f"      ACC={b['acc']:.2f}%, REC={b['recall']:.2f}%, PRE={b['precision']:.2f}%, F1={b['f1']:.2f}%")

    print(f"\n    [各风险类别] 二分类指标:")
    print(f"    {'类别':<50} {'总数':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'ACC':>8} {'REC':>8} {'PRE':>8} {'F1':>8}")
    print("    " + "-" * 120)
    for cat in all_categories:
        s = per_cat_m[cat]
        print(f"    {cat:<50} {s['total']:>6} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6} {s['tn']:>6} {s['acc']:>7.2f}% {s['recall']:>7.2f}% {s['precision']:>7.2f}% {s['f1']:>7.2f}%")

    report = {
        "input_file": input_path,
        "label_file": label_path,
        "dataset_size": dataset_size,
        "source_distribution": {
            src: {
                "count": cnt,
                "ratio": cnt / dataset_size * 100,
                "harmful": per_source_label[src]["harmful"],
                "safe": per_source_label[src]["safe"],
            }
            for src, cnt in source_counter.items()
        },
        "label_distribution": dict(label_counter),
        "category_distribution": {
            cat: {
                "count": cnt,
                "ratio": cnt / dataset_size * 100,
                "harmful": per_cat_harmful.get(cat, 0),
                "harmful_ratio": (per_cat_harmful.get(cat, 0) / cnt * 100) if cnt else 0,
            }
            for cat, cnt in category_counter.items()
        },
        "inference": {
            "matched": matched,
            "parse_fail_treated_as_safe": parse_fail,
            "binary": binary_m,
            "per_category": per_cat_m,
        },
    }

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[6] 详细统计报告已保存至: {output_path}")
    print("=" * 70)
    print("分析完成。")


if __name__ == "__main__":
    main()
