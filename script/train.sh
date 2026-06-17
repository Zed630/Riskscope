CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1605632 \
TRL_EXPERIMENTAL_SILENCE=1 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model /dataset-2041446603085574144/Riskscope/model/Qwen3-VL-8B-Instruct \
    --external_plugins /dataset-2041446603085574144/Riskscope/script/safeReward.py \
    --reward_funcs safe_reward \
    --use_vllm true \
    --vllm_mode colocate \
    --enable_thinking true \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 4 \
    --tuner_type lora \
    --torch_dtype float16 \
    --dataset /dataset-2041446603085574144/Riskscope/test_data/Safety/all_dataset_1.0.1/train_data.jsonl \
    --val_dataset /dataset-2041446603085574144/Riskscope/test_data/Safety/all_dataset_1.0.1/val_data.jsonl \
    --max_completion_length 4096 \
    --max_model_len 40000 \
    --sleep_level 1 \
    --num_train_epochs 2 \
    --dataloader_num_workers 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 100 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir /dataset-2041446603085574144/Riskscope/outputs/train_output \
    --warmup_steps 100 \
    --num_generations 4 \
    --generation_batch_size 4 \
    --temperature 1.0 \
    --log_completions true \
    --beta 0.001 \