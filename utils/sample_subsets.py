import json
import os
import random

DATASET_PATH = "/dataset-2041446603085574144/Riskscope/test_data/test_data.jsonl"
OUTPUT_DIR = "/dataset-2041446603085574144/Riskscope/outputs/random_subsets"
SEED = 42
SAMPLE_SIZE = 5000
N_TRIALS = 3


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
    return is_harmful, []


def main():
    print("=" * 70)
    print("数据集抽样 (生成 subset_1/2/3.jsonl)")
    print("=" * 70)

    dataset = load_jsonl(DATASET_PATH)
    print(f"\n数据集总条目数: {len(dataset)}")

    # 按原 analyze_dataset.py 的逻辑过滤出 valid items:
    # 只有 solution 中 gt_harmful 不是 None 的条目才参与抽样
    valid_items = []
    for item in dataset:
        gt_harmful, _ = parse_solution(item.get("solution"))
        if gt_harmful is not None:
            valid_items.append(item)

    n_total = len(valid_items)
    print(f"可抽样条目数 (solution 有效): {n_total}")

    if n_total < SAMPLE_SIZE:
        print(f"[ERROR] 可抽样条目数 {n_total} < 每次采样数量 {SAMPLE_SIZE}")
        return

    random.seed(SEED)

    for trial in range(N_TRIALS):
        sample_indices = random.sample(range(n_total), SAMPLE_SIZE)
        sample_items = [valid_items[i] for i in sample_indices]

        out_name = f"subset_{trial + 1}.jsonl"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for item in sample_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\n第 {trial + 1} 次采样 -> {out_name} ({len(sample_items)} 条)")
        print(f"  已保存: {out_path}")

    print("\n" + "=" * 70)
    print("抽样完成。")


if __name__ == "__main__":
    main()
