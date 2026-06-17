PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --adapters /dataset-2041446603085574144/Riskscope/model/v7-20260607-003145/checkpoint-2612 \
    --max_new_tokens 4096 \
    --torch_dtype float16 \
    --device_map auto \
    --load_data_args false \
    --val_dataset /dataset-2041446603085574144/Riskscope/outputs/random_subsets/subset_${1}.jsonl \
    --result_path /dataset-2041446603085574144/Riskscope/outputs/random_subsets/infer_${1}.jsonl \
    --max_batch_size 4