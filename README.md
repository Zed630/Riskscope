# Riskscope

基于 Qwen3-VL-8B-Instruct 的视觉风险内容检测项目，使用 GRPO (Group Relative Policy Optimization) 强化学习方法进行训练。

## 项目简介

Riskscope 是一个多模态视觉模型，用于检测图片中的有害/安全内容。项目采用 LoRA 微调 + GRPO 强化学习的训练策略，通过奖励模型 `safeReward.py` 引导模型生成正确的风险判定结果。

## 目录结构

```
Riskscope/
├── images/
├── outputs/
├── script/
│   ├── sample_subsets.sh      # 采样子集脚本
│   ├── infer.sh               # 模型推理脚本
│   ├── run_subset_infer.sh    # 后台推理脚本
│   ├── analysis_subset.sh     # 结果分析脚本
│   ├── train.sh               # 模型训练脚本
│   └── safeReward.py          # 奖励模型
├── utils/
│   ├── sample_subsets.py      # 数据采样工具
│   ├── evaluate_subset.py     # 子集评估工具
│   └── evaluate.py            # 通用评估工具
├── test_data/
│   ├── Safety/
│   │   └── all_dataset_1.0.1/ # 训练数据集
│   │       ├── train_data.jsonl
│   │       ├── val_data.jsonl
│   │       └── test_data.jsonl
│   └── test_data.jsonl        # 全集测试数据（用于推理评估）
└── model/                     # 模型文件（位于服务器）
    └── Qwen3-VL-8B-Instruct/  # 基座模型
    ├── v7-20260607-003145/    # Adapter 模型文件
```

## 环境要求

- **框架**: [swift](https://github.com/modelscope/ms-swift) (ModelScope 多模态训练框架)
- **基座模型**: Qwen3-VL-8B-Instruct
- **硬件**: 4 × NVIDIA GPU

> **注意**: images 和 model 文件均位于服务器路径 `/dataset-2041446603085574144/Riskscope/` 下，所有脚本默认使用该绝对路径。

## 使用流程

### 流程一：子集采样 → 子集推理 → 子集分析

适合对测试集进行多次随机采样，评估模型在不同数据分布下的稳定性。

**第 1 步：生成 N 批采样数据**

```bash
bash /dataset-2041446603085574144/Riskscope/script/sample_subsets.sh
```

该脚本从 `test_data/test_data.jsonl` 中：
- 每次随机采样 **5000 条** 有效样本（solution 字段可解析）
- 默认生成 **3 批** 独立子集，种子固定为 42
- 输出至 `/dataset-2041446603085574144/Riskscope/outputs/random_subsets/subset_{1,2,3}.jsonl`

**第 2 步：对每个子集进行推理**

后台运行推理（推荐使用 nohup 避免会话中断）：

```bash
bash /dataset-2041446603085574144/Riskscope/script/run_subset_infer.sh 1
bash /dataset-2041446603085574144/Riskscope/script/run_subset_infer.sh 2
bash /dataset-2041446603085574144/Riskscope/script/run_subset_infer.sh 3
```

参数 `1`/`2`/`3` 对应子集编号。推理结果保存为 `infer_{n}.jsonl`。

**第 3 步：分析每批结果**

```bash
bash /dataset-2041446603085574144/Riskscope/script/analysis_subset.sh 1
bash /dataset-2041446603085574144/Riskscope/script/analysis_subset.sh 2
bash /dataset-2041446603085574144/Riskscope/script/analysis_subset.sh 3
```

分析报告输出：
- JSON: `outputs/random_subsets/subset_{n}_report.json`
- CSV: `outputs/random_subsets/subset_{n}_report.csv`

报告包含：
- 整体二分类指标（ACC / Recall / Precision / F1）
- 各风险类别的二分类指标
- 数据来源构成分析

---

### 流程二：子集采样 → 全集推理 → 子集分析

适合先对全集数据做一次推理，然后复用全集推理结果在各子集上评估，避免重复推理。

**第 1 步：生成 N 批采样数据**（同流程一）

```bash
bash /dataset-2041446603085574144/Riskscope/script/sample_subsets.sh
```

**第 2 步：跑全集推理**

修改 `infer.sh` 中的数据集参数指向全集：

```bash
# 手动调用 swift infer，将 val_dataset 替换为全集路径
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --adapters /dataset-2041446603085574144/Riskscope/model/v7-20260607-003145/checkpoint-2612 \
    --max_new_tokens 4096 \
    --torch_dtype float16 \
    --device_map auto \
    --load_data_args false \
    --val_dataset /dataset-2041446603085574144/Riskscope/test_data/test_data.jsonl \
    --result_path /dataset-2041446603085574144/Riskscope/outputs/random_subsets/infer_full.jsonl \
    --max_batch_size 4
```

**第 3 步：复用全集推理结果分析子集**

将 `analysis_subset.sh` 中的 `--infer` 参数替换为全集推理结果路径，例如：

```bash
python /dataset-2041446603085574144/Riskscope/utils/evaluate_subset.py \
  --subset "/dataset-2041446603085574144/Riskscope/outputs/random_subsets/subset_1.jsonl" \
  --infer /dataset-2041446603085574144/Riskscope/outputs/random_subsets/infer_full.jsonl \
  --output "/dataset-2041446603085574144/Riskscope/outputs/random_subsets/subset_1_report_full.json"
```

`evaluate_subset.py` 会根据 **图片路径 + 用户 query** 作为 key 在全集中匹配对应的 response，因此即使在子集中也能拿到推理结果。

---

## 模型训练

训练脚本在 `script/train.sh`，核心训练参数如下：

| 参数 | 值 |
|------|-----|
| 训练方法 | GRPO (Group Relative Policy Optimization) |
| 基座模型 | `/dataset-2041446603085574144/Riskscope/model/Qwen3-VL-8B-Instruct` |
| 微调方式 | LoRA |
| 精度 | float16 |
| 推理加速 | vLLM (colocate 模式, 4 卡张量并行) |
| 奖励函数 | `safeReward.py` 中的 `safe_reward` |
| 数据集 | `test_data/Safety/all_dataset_1.0.1/train_data.jsonl` |
| 验证集 | `test_data/Safety/all_dataset_1.0.1/val_data.jsonl` |
| 训练轮数 | 2 epochs |
| 学习率 | 1e-4 |
| 最大生成长度 | 4096 tokens |
| 梯度累积步数 | 4 |
| 思考模式 | enable_thinking=true |
| 输出目录 | `outputs/train_output` |

**启动训练：**

```bash
bash /dataset-2041446603085574144/Riskscope/script/train.sh
```

训练完成后，checkpoint 将保存在 `outputs/train_output` 目录下，可将其路径填入 `infer.sh` 的 `--adapters` 参数用于推理。

## 评估指标说明

`evaluate_subset.py` 计算以下二分类指标：

- **Accuracy (ACC)**: 整体准确率
- **Recall (REC)**: 有害样本中被正确识别为有害的比例（查全率）
- **Precision (PRE)**: 被识别为有害的样本中真正有害的比例（查准率）
- **F1 Score**: Recall 和 Precision 的调和平均

同时按各风险类别（如暴力、色情、歧视等）分别计算上述指标，便于定位模型在特定风险类型上的表现。

## 数据格式

### 训练/推理数据 (JSONL)

每行一条 JSON，关键字段：

- `images`: 图片路径数组 (路径需要转换)，完整的开源数据集可在 https://modelscope.cn/datasets/zed6666/Riskscope/files 查看。其他业务数据需要找孙诗奇获取。
- `messages`: 对话历史（包含 user query 和 assistant response）
- `solution`: 标注答案，格式为 `"harmful, [\"类别1\", \"类别2\"]"` 或 `"safe, []"`

### 推理输出 (JSONL)

每行一条 JSON，关键字段：

- `images`: 图片路径（用于匹配）
- `messages`: 用户 query（用于匹配）
- `response`: 模型生成的回答，包含 `is_harmful` 和 `risk_categories`
