# train.sh
cd /workspace/LLaMA-Factory
export HF_TOKEN=YOUR_HF_TOKEN
deepspeed --num_gpus 8 src/train_bash.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage orpo \
    --do_train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset dpo_mix_en,dpo_mix_zh \
    --template llama3 \
    --finetuning_type full \
    --output_dir /opt/ml/checkpoints \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --log_level info \
    --logging_steps 5 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --save_steps 100 \
    --learning_rate 5e-6 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --do_eval false \
    --max_steps -1 \
    --bf16 true \
    --seed 42 \
    --warmup_ratio 0.1 \
    --cutoff_len 8192 \
    --flash_attn auto \
    --orpo_beta 0.05 \
    --optim paged_adamw_32bit \
    --overwrite_output_dir 