export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
# export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
# echo "WANDB_ID is $WANDB_RUN_ID"

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
  
tensorboard --logdir=./tb_logs --port=6006 &
echo "Training completed!"
echo "启动 Tensorboard 服务..."
echo "请访问 http://localhost:6006 查看训练进度"