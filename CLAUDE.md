# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a research codebase for **DIFUSCO** (Diffusion-based Combinatorial Optimization) with cross-problem reinforcement learning training. The codebase implements diffusion models for solving combinatorial optimization problems, particularly:

- **TSP (Traveling Salesman Problem)**: Graph-based diffusion models for tour optimization
- **VRP variants**: Vehicle routing problems including CVRP, VRPTW, OVRP, VRPB, VRPL and their combinations
- **MIS (Maximum Independent Set)**: Graph-based diffusion models for set optimization
- **Cross-problem training**: Multi-task learning approach for routing problems with zero-shot generalization

### Core Components

- **`difusco/train.py`**: Main training entry point with Lightning trainer setup
- **`difusco/pl_tsp_model.py`**: Lightning module for TSP with diffusion and RL components
- **`difusco/pl_mis_model.py`**: Lightning module for MIS problems  
- **`difusco/pl_meta_model.py`**: Meta-learning module for cross-problem scenarios
- **`difusco/co_datasets/`**: Dataset implementations for TSP, VRP, and MIS
- **`difusco/models/`**: Neural network architectures (GNN encoders, diffusion models)
- **`difusco/utils/`**: Utility functions for diffusion schedulers, TSP/MIS operations

### Architecture Patterns

- **Lightning-based training**: All models inherit from `pytorch_lightning.LightningModule`
- **Diffusion framework**: Implements both categorical and Gaussian diffusion types
- **Reinforcement learning integration**: POMO-based policy optimization with diffusion guidance
- **Multi-task learning**: Hybrid training across different problem types
- **Cross-problem generalization**: Unified models that can handle multiple VRP variants

## Training Commands

### Basic Training

```bash
# TSP diffusion training
python -u difusco/train.py \
  --task "tsp" \
  --diffusion_type "categorical" \
  --do_train \
  --learning_rate 0.0002 \
  --batch_size 64 \
  --num_epochs 50

# Cross-problem VRP training  
./training_hybrid_50.sh CVRP VRPL
```

### Environment Setup

```bash
# Set Python path
export PYTHONPATH="$PWD:$PYTHONPATH"

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Activate conda environment
conda activate difusco_basic_py39
```

### Available Training Scripts

- `training_tsp_50.sh`: TSP-50 training
- `training_hybrid_50.sh`: Cross-problem VRP training
- `training_cvrp_50.sh`: CVRP-specific training
- `training_ovrpbltw_50.sh`: OVRPBLTW training
- `test_tsp_50.sh`: TSP evaluation

## Key Configuration Parameters

### Diffusion Settings
- `--diffusion_type`: "categorical" or "gaussian"
- `--inference_schedule`: "cosine" or "linear"
- `--inference_diffusion_steps`: Number of diffusion steps (default: 50)

### RL Training Parameters
- `--use_pomo`: Enable POMO-based reinforcement learning
- `--rl_loss_weight`: Weight for RL loss component (default: 0.01)
- `--rl_compute_frequency`: Frequency of RL loss computation (default: 10)
- `--pomo_temperature`: Temperature for POMO solver (default: 1.0)

### Numerical Stability
- `--max_logit_value`: Maximum logit value to prevent softmax overflow (default: 50.0)
- `--min_prob_value`: Minimum probability value to prevent log(0) (default: 1e-8)
- `--max_advantage`: Maximum advantage value to prevent gradient explosion (default: 20.0)

## Data Paths

- TSP data: `data/tsp/tsp50_train_concorde.txt`
- VRP data: `data/vrp/CVRP50.pkl`, `data/vrp/VRPL50.pkl`, etc.
- MIS data: Various `.gpickle` files for different graph types

## Problem Types

### Supported VRP Variants
CVRP, VRPTW, OVRP, VRPB, VRPL, OVRPTW, OVRPB, VRPBL, VRPBTW, VRPLTW, OVRPBL, OVRPBTW, OVRPLTW, VRPBLTW, OVRPBLTW

### Task Types
- `"tsp"`: Traveling Salesman Problem
- `"mis"`: Maximum Independent Set
- `"hybrid"`: Cross-problem multi-task learning

## Testing and Evaluation

### Running Tests
```bash
# Test TSP-50 with trained model
./test_tsp_50.sh

# Test specific VRP problems  
python -u difusco/train.py \
  --task "hybrid" \
  --do_test \
  --ckpt_path "./tb_logs/[experiment_name]/checkpoints/last.ckpt"
```

### Debug and Analysis Scripts
- `debug_rl_loss.py`: RL loss debugging utilities
- `test_*.py`: Various testing and validation scripts
- `optimizer_matrix_demo.py`: Optimization matrix demonstrations

## Multi-Task Neural Combinatorial Optimization (MTNCO)

The codebase includes MTNCO baseline implementations in `MTNCO/`:
- **MTPOMO/**: Multi-task POMO implementation for cross-problem training
- **Baseline/**: Single-task baselines for CVRP, OVRP, VRPB, VRPL, VRPTW
- **Trained_models/**: Pre-trained models for problem sizes 50 and 100
- **Test_instances/**: Comprehensive test data for 11 VRP variants

### MTNCO Commands
```bash
cd MTNCO/MTPOMO/POMO/
python train_n50.py    # Multi-task training
python test_n50.py     # Testing
```

## Monitoring and Logging

### TensorBoard Integration
```bash
# Start TensorBoard server
tensorboard --logdir=./tb_logs

# Access at http://localhost:6006
```

### Log Structure
- Training logs: `tb_logs/[experiment_name]/`
- Route visualizations: Auto-generated PNG comparisons in experiment directories
- Gradient monitoring: Built-in gradient norm tracking via `GradientLoggingCallback`

## Development Notes

- **Numerical Stability**: Extensive safeguards for RL training including gradient clipping, logit clamping, and error recovery
- **Cross-problem Generalization**: Unified models handle multiple VRP variants with zero-shot transfer
- **Hybrid Training**: Supports simultaneous training on multiple problem types
- **Debug Modes**: Comprehensive debugging flags for numerical issue diagnosis (`rl_debug`, `test_debug`)
- **Visualization**: Automatic route comparison plotting during validation