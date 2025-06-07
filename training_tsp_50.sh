#!/bin/bash

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u difusco/train.py \
  --task "tsp" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "./" \
  --training_split "data/tsp/tsp50_train_concorde.txt" \
  --validation_split "data/tsp/tsp50_test_concorde.txt" \
  --test_split "data/tsp/tsp50_test_concorde.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50

# 检查tensorboard是否已在运行，如果是则停止
pkill -f "tensorboard.*port=6006" 2>/dev/null

# 启动tensorboard
tensorboard --logdir=./tb_logs --port=6006 &

echo "Training completed!"
echo "启动 Tensorboard 服务..."
echo "请访问 http://localhost:6006 查看训练进度"