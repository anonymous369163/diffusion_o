#!/bin/bash

# Example script for training with TensorBoard logging
# This replaces the previous wandb-based training approach

# Set training parameters
TASK="tsp"
STORAGE_PATH="./experiments"
PROJECT_NAME="tsp_diffusion_tensorboard"
LOGGER_NAME="tsp50_experiment"

# Create storage directory if it doesn't exist
mkdir -p $STORAGE_PATH

# Run training with TensorBoard logging
python difusco/train.py \
  --task $TASK \
  --storage_path $STORAGE_PATH \
  --project_name $PROJECT_NAME \
  --logger_name $LOGGER_NAME \
  --training_split "data/tsp/tsp50_train_concorde.txt" \
  --validation_split "data/tsp/tsp50_test_concorde.txt" \
  --test_split "data/tsp/tsp50_test_concorde.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --learning_rate 1e-4 \
  --diffusion_steps 1000 \
  --n_layers 12 \
  --hidden_dim 256 \
  --do_train \
  --do_test

echo "Training completed!"
echo "To view TensorBoard logs, run:"
echo "tensorboard --logdir=$STORAGE_PATH/tb_logs"
echo "Then open http://localhost:6006 in your browser" 