#!/bin/bash
# 激活正确的conda环境
# 初始化conda
eval "$(conda shell.bash hook)"

conda activate difusco_basic_py39

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "$(dirname "$0")"

# 你可以在这里修改要使用的问题类型
# 支持的问题类型: CVRP VRPTW OVRP VRPB VRPL OVRPTW OVRPB VRPBL VRPBTW VRPLTW OVRPBL OVRPBTW OVRPLTW VRPBLTW OVRPBLTW
PROBLEM_TYPES="CVRP VRPTW OVRP VRPB VRPL"

# 如果通过命令行参数指定了问题类型，则使用命令行参数
if [ $# -gt 0 ]; then
    PROBLEM_TYPES="$@"
    echo "使用命令行指定的问题类型: $PROBLEM_TYPES"
else
    PROBLEM_TYPES="$PROBLEM_TYPES"
    echo "使用默认的问题类型组合: $PROBLEM_TYPES"
fi

EXPERIMENT_NAME="hybrid_$(echo $PROBLEM_TYPES | tr ' ' '_')_$(date +%m%d_%H%M)"

echo "开始混合训练实验: $EXPERIMENT_NAME"
echo "使用的问题类型: $PROBLEM_TYPES"

python -u difusco/train.py \
  --task "hybrid" \
  --problem_type "hybrid" \
  --hybrid_problem_types $PROBLEM_TYPES \
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
  --rl_compute_frequency 10 \
  --use_pomo \
  --rl_loss_weight 0.01 \
  --rl_baseline_decay 0.95 \
  --pomo_temperature 1 \
  --do_train \
  --do_test \
  --no_debug \
  --add_prior \
  --logger_name $EXPERIMENT_NAME \
  --training_split "data/vrp/CVRP50.pkl" \
  --validation_split "data/vrp/CVRP50.pkl" \
  --test_split "data/vrp/CVRP50.pkl" \
  --draw_route_comparison 

echo "训练完成！"
echo "实验名称: $EXPERIMENT_NAME"
echo "日志位置: ./tb_logs/${EXPERIMENT_NAME}_train/"

# 所有支持的问题类型列表（供参考）:
# ["CVRP", "VRPTW", "OVRP", "VRPB","VRPL", "OVRPTW", "OVRPB", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]

# tensorboard --logdir=./tb_logs &
# echo "Training completed!"
# echo "启动 Tensorboard 服务..."
# echo "请访问 http://localhost:6006 查看训练进度"
