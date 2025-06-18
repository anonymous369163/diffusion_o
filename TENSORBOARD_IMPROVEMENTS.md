# TensorBoard 超参数和指标记录改进

本次改进为DIFUSCO项目的TensorBoard日志记录添加了全面的超参数和训练指标记录功能。

## 主要改进

### 1. 超参数记录 (Hyperparameters)
在模型初始化时自动记录所有重要的超参数到TensorBoard：

#### 模型架构参数
- `n_layers`: GNN层数
- `hidden_dim`: 隐藏层维度
- `aggregation`: 聚合方式
- `sparse_factor`: 稀疏因子

#### 训练参数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `weight_decay`: 权重衰减
- `lr_scheduler`: 学习率调度器
- `num_epochs`: 训练轮数

#### 扩散模型参数
- `diffusion_type`: 扩散模型类型 (categorical/gaussian)
- `diffusion_schedule`: 扩散调度 (cosine/linear)
- `diffusion_steps`: 扩散步数
- `inference_diffusion_steps`: 推理扩散步数
- `inference_schedule`: 推理调度
- `inference_trick`: 推理技巧 (ddim等)

#### 采样参数
- `sequential_sampling`: 序列采样次数
- `parallel_sampling`: 并行采样次数

#### 强化学习参数 (如果启用)
- `rl_loss_weight`: 强化学习损失权重
- `rl_baseline_decay`: 基线衰减率
- `rl_compute_frequency`: RL计算频率
- `use_pomo`: 是否使用POMO方法

### 2. 训练过程记录

#### 基本训练指标
- `train/ce_loss`: 交叉熵损失 (categorical)
- `train/mse_loss`: 均方误差损失 (gaussian)
- `train/total_loss`: 总损失 (包含RL损失)
- `train/rl_loss`: 强化学习损失
- `train/gradient_norm`: 梯度范数
- `train/learning_rate`: 当前学习率

#### 强化学习指标 (如果启用)
- `train/avg_reward`: 平均奖励
- `train/best_reward`: 最佳奖励
- `train/baseline`: 基线值
- `train/avg_pred_cost`: 平均预测成本
- `train/avg_gt_cost`: 平均真实成本
- `train/cost_gap_percent`: 成本差距百分比

#### 训练过程信息
- `train/batch_idx`: 当前批次索引
- `train/batch_size`: 当前批次大小
- `train/avg_diffusion_timestep`: 平均扩散时间步

### 3. 验证和测试指标

#### TSP任务
- `val/solved_cost`, `test/solved_cost`: 求解成本
- `val/gt_cost`, `test/gt_cost`: 真实最优成本
- `val/gap_percentage`, `test/gap_percentage`: 与最优解的差距百分比
- `val/avg_gap_percentage`, `test/avg_gap_percentage`: 平均差距百分比
- `val/std_gap_percentage`, `test/std_gap_percentage`: 差距标准差
- `val/2opt_iterations`, `test/2opt_iterations`: 2-opt迭代次数
- `val/merge_iterations`, `test/merge_iterations`: 合并迭代次数

#### MIS任务
- `val/solved_cost`, `test/solved_cost`: 求解成本 (最大独立集大小)
- `val/gt_cost`, `test/gt_cost`: 真实最优成本
- `val/avg_solved_cost`, `test/avg_solved_cost`: 平均求解成本
- `val/std_solved_cost`, `test/std_solved_cost`: 求解成本标准差
- `val/gap_percentage`, `test/gap_percentage`: 与最优解的差距百分比
- `val/num_nodes`, `test/num_nodes`: 图中节点数量

### 4. 模型和数据集信息记录

在训练开始时记录：
- **模型信息**: 总参数数量、可训练参数数量、模型架构描述
- **数据集信息**: 训练、验证、测试数据集路径
- **训练配置**: 任务类型、批次大小、学习率等配置的文本描述
- **扩散配置**: 扩散模型相关参数的文本描述

### 5. 梯度监控

通过自定义回调函数 `GradientLoggingCallback` 记录：
- `train/gradient_norm`: 总梯度范数
- `train/gradient_norm_avg`: 平均梯度范数
- `train/learning_rate`: 每个epoch的学习率

## 使用方法

这些改进已经集成到现有的训练脚本中，使用时无需额外配置：

```bash
python difusco/train.py [your_arguments]
```

训练过程中，所有指标将自动记录到TensorBoard日志中。

## 查看结果

启动TensorBoard查看记录的指标：

```bash
tensorboard --logdir=./tb_logs
```

在TensorBoard界面中可以找到：
- **SCALARS**: 所有数值指标的时间序列图
- **HPARAMS**: 超参数表格和超参数与性能指标的关系图
- **TEXT**: 模型架构和配置的文本描述

## 注意事项

1. 所有超参数在模型初始化时自动保存
2. 训练指标支持同时记录step级别和epoch级别的值
3. 验证和测试指标在每个epoch结束时记录
4. 梯度监控在每个优化步骤前执行
5. 强化学习相关指标仅在启用RL损失时记录

这些改进大大增强了训练过程的可观测性，便于分析模型性能和调试训练过程。 