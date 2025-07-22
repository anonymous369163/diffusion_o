#!/bin/bash
# 单案例CVRP50训练脚本 - 用于测试算法在单个案例上的学习能力
# 激活正确的conda环境
# 初始化conda
eval "$(conda shell.bash hook)"

conda activate difusco_basic_py39

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "$(dirname "$0")"

echo "=========================================="
echo "单案例CVRP50训练实验"
echo "目标：测试算法在单个案例上能否学出比真实值更好的方案"
echo "=========================================="

python -u difusco/train.py \
  --task "cvrp" \
  --diffusion_type "categorical" \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "./" \
  --batch_size 64 \
  --num_epochs 100 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --rl_compute_frequency 5 \
  --use_pomo \
  --rl_loss_weight 0.00 \
  --rl_baseline_decay 0.95 \
  --pomo_temperature 1 \
  --do_train \
  --do_test \
  --no_debug \
  --add_prior \
  --logger_name "cvrp_single_case_experiment" \
  --problem_type "CVRP" \
  --training_split "data/vrp/CVRP50.pkl" \
  --validation_split "data/vrp/CVRP50.pkl" \
  --test_split "data/vrp/CVRP50.pkl" \
  --draw_route_comparison \
  --single_case_mode \
  --single_case_copies 2000 \
  --n_layers 12 \
  --hidden_dim 256

echo "单案例训练完成！"
echo "日志保存在: ./tb_logs/cvrp_single_case_experiment_train/"
echo "可以使用以下命令查看训练进度:"
echo "tensorboard --logdir=./tb_logs/cvrp_single_case_experiment_train/" 