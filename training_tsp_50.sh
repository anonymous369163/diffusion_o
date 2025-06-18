#!/bin/bash
# 激活conda环境
# 添加执行权限
source /yuepeng/code/conda-envs/bin/activate

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "$(dirname "$0")"

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
  --num_epochs 100 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --rl_compute_frequency 1 \
  --use_pomo \
  --rl_loss_weight 0.01 \
  --rl_baseline_decay 0.95 \
  --pomo_temperature 1 \
  --do_train \
  --do_test 
  
# tensorboard --logdir=./tb_logs &
# echo "Training completed!"
# echo "启动 Tensorboard 服务..."
# echo "请访问 http://localhost:6006 查看训练进度"