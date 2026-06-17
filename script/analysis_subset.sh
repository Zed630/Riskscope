python /dataset-2041446603085574144/Riskscope/utils/evaluate_subset.py \
  --subset "/dataset-2041446603085574144/Riskscope/outputs/random_subsets/subset_${1}.jsonl" \
  --infer /dataset-2041446603085574144/Riskscope/model/v7-20260607-003145/checkpoint-2612/infer_result/infer_${1}.jsonl \
  --output "/dataset-2041446603085574144/Riskscope/outputs/random_subsets/subset_${1}_report.json"