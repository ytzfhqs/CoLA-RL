#!/bin/bash

set -e

# 环境变量
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY="a71c944c7ae6d25b6018822b34e54c04b010809b"
export PYTHONPATH="/root/workspace/qingsong/cola_rl"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="0,3"

# 配置参数
use_dynamic_bsz="True"
actor_ppo_max_token_len=$((1024 * 3))
infer_ppo_max_token_len=$((1024 * 3))

sp_size=2
gpu_nums=2

checkpoint_path="model/Qwen3-1.7B"
project_name="CoLA"
experiment_name="Qwen3-1.7B-remax"

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.path="${checkpoint_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.ref.strategy="fsdp2" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${sp_size}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.name="vllm" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    algorithm.adv_estimator="remax" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_penalty="kl" \
    algorithm.use_kl_in_reward=True \
    critic.strategy="fsdp2" \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=256 \
    data.max_response_length=2048 \
    data.train_batch_size=256 \
    data.train_files="data/cola_rl/train.parquet" \
    data.truncation="error" \
    data.val_files="data/cola_rl/test.parquet" \
    reward_model.strategy="fsdp2" \
    trainer.critic_warmup=0 \
    trainer.experiment_name="${experiment_name}" \
    trainer.logger="wandb" \
    trainer.n_gpus_per_node="${gpu_nums}" \
    trainer.nnodes=1 \
    trainer.project_name="${project_name}" \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False \
    "$@"
