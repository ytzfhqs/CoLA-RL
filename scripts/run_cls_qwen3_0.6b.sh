export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0

python3 -m train_cls \
    --model_path "model/Qwen3-0.6B" \
    --num_train_epochs 3 \
    --train_batch_size 8 \
    --eval_batch_size 16
