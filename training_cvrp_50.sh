#!/bin/bash
# 激活正确的conda环境
# 初始化conda
eval "$(conda shell.bash hook)"

conda activate difusco_39

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
cd "$(dirname "$0")"

python -u difusco/train.py \
  --task "vrp" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "./" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --rl_compute_frequency 10 \
  --use_pomo \
  --rl_loss_weight 0.0 \
  --rl_baseline_decay 0.95 \
  --pomo_temperature 1 \
  --do_train \
  --do_test \
  --no_debug \
  --logger_name "cvrp_50_rl_0.0" \
  --problem_type "CVRP" \
  --training_split "data/vrp/CVRP.pkl" \
  --validation_split "data/vrp/CVRP.pkl" \
  --test_split "data/vrp/CVRP.pkl" \

# ["TSP", "CVRP", "OVRP", "VRPB","VRPL", "VRPTW", "OVRPTW", "OVRPB", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]
# tensorboard --logdir=./tb_logs &
# echo "Training completed!"
# echo "启动 Tensorboard 服务..."
# echo "请访问 http://localhost:6006 查看训练进度"
