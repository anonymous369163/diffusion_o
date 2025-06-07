# 扩散模型 + 强化学习 TSP训练器

本项目实现了结合扩散模型和强化学习的TSP求解方法，**以扩散模型生成的邻接矩阵为主要决策依据**，可选地引入简单神经网络进行增强。

## 项目结构

```
├── diffusion_rl_trainer.py        # 主要训练器（以邻接矩阵为主）
├── test_different_configs.py      # 配置对比测试脚本
└── README_扩散RL训练.md           # 使用说明文档
```

## 核心创新 🎯

1. **邻接矩阵主导决策**: 扩散模型生成的邻接矩阵是动作选择的主要依据
2. **可选神经网络增强**: 用户可以选择不使用、简单线性层或MLP来增强决策
3. **端到端训练**: 将求解质量的奖励反传回扩散模型，实现联合优化
4. **灵活配置**: 通过超参数控制模型复杂度

## 决策机制

### 纯邻接矩阵决策 (推荐)
```python
# 直接使用扩散模型生成的邻接矩阵进行决策
logits = adj_matrix[batch_idx, current_node]  # 从当前节点到所有节点的连接权重
```

### 线性层增强
```python
# 邻接矩阵 + 简单特征相似度
logits = 0.8 * adj_logits + 0.2 * feature_similarity
```

### MLP增强
```python
# 邻接矩阵 + 多层感知机特征提取
logits = 0.8 * adj_logits + 0.2 * mlp_features
```

## 使用方法

### 快速开始

1. **快速测试纯邻接矩阵决策**:
   ```bash
   python test_different_configs.py --mode quick
   ```

2. **完整配置对比测试**:
   ```bash
   python test_different_configs.py --mode full
   ```

3. **直接训练（推荐配置）**:
   ```bash
   python diffusion_rl_trainer.py
   ```

### 配置选项

在 `diffusion_rl_trainer.py` 的 `main()` 函数中可以调整以下配置：

```python
# 配置选项
use_neural_network = True     # 是否使用额外的神经网络
network_type = 'linear'       # 网络类型: 'linear', 'mlp', 'none'

# 'linear': 简单线性层增强（推荐）
# 'mlp': 多层感知机增强（复杂问题）
# 'none': 纯邻接矩阵决策（最简单、最快）
```

### 训练参数

```python
trainer = DiffusionRLTrainer(
    diffusion_model_path='path/to/checkpoint.ckpt',  # 扩散模型检查点
    num_nodes=50,               # TSP问题节点数
    batch_size=32,              # 批次大小
    pomo_size=50,               # POMO并行数量
    lr=1e-4,                    # 学习率
    device='cuda',              # 设备
    # 新增配置
    use_neural_network=True,    # 是否使用神经网络
    network_type='linear',      # 网络类型
    embedding_dim=64            # 嵌入维度
)
```

## 技术特点

### 🔥 邻接矩阵主导的决策机制

**核心优势**:
- **简单高效**: 直接利用扩散模型的输出，减少额外计算
- **可解释性强**: 决策过程透明，基于节点间连接概率
- **训练稳定**: 避免复杂网络结构带来的训练困难

**决策流程**:
1. 扩散模型根据节点坐标生成邻接矩阵 `A`
2. 从当前节点 `i` 到所有节点的权重: `logits = A[i, :]`
3. 应用访问掩码，使用softmax得到概率分布
4. 采样或贪心选择下一个节点

### 🛠️ 可选的神经网络增强

**三种配置对比**:

| 配置 | 参数量 | 训练速度 | 表达能力 | 适用场景 |
|------|--------|----------|----------|----------|
| 纯邻接矩阵 | 最少 | 最快 | 中等 | 大多数情况 |
| 线性层增强 | 少 | 快 | 中上 | 需要轻量增强 |
| MLP增强 | 中等 | 中等 | 强 | 复杂问题 |

### 📊 POMO强化学习

- **并行探索**: 同时从多个起始节点构建路径
- **多样性**: 增加解的多样性，提高最优解概率
- **效率**: 单次前向传播获得多个候选解

## 实验和评估

### 配置对比实验

运行以下命令进行自动对比：

```bash
# 完整对比测试（推荐）
python test_different_configs.py --mode full

# 快速测试
python test_different_configs.py --mode quick
```

实验将自动输出类似以下结果：

```
最终对比结果:
============================================================
纯邻接矩阵决策        : 3.2451
线性层增强           : 3.2389
MLP增强             : 3.2567
============================================================
最佳配置: 线性层增强 (平均路径长度: 3.2389)
```

### 性能监控

训练过程中监控以下指标：

```
Epoch   10: 平均路径长度=3.2451, 总损失=0.1234, RL损失=0.1123, 扩散损失=0.0111
```

- **平均路径长度**: 越小越好
- **RL损失**: 强化学习模型的损失
- **扩散损失**: 扩散模型的反馈损失

## 优势分析

### 🎯 相比传统方法的优势

1. **端到端优化**: 扩散模型和求解器联合训练
2. **先验知识**: 扩散模型提供问题特定的先验知识
3. **灵活性**: 可以适应不同复杂度的需求

### 🚀 相比复杂神经网络的优势

1. **训练稳定**: 避免复杂网络的训练困难
2. **计算高效**: 减少前向传播时间
3. **内存友好**: 参数量少，内存占用小
4. **可解释**: 决策过程基于邻接矩阵，易于理解

## 扩展和定制

### 自定义决策权重

```python
# 在TSPRLModel.forward()中调整权重
alpha = 0.9  # 增加邻接矩阵权重
beta = 0.1   # 减少特征权重
logits = alpha * adj_logits + beta * feature_similarity
```

### 添加新的网络类型

```python
# 在TSPRLModel.__init__()中添加新类型
elif self.network_type == 'attention':
    self.attention_layer = nn.MultiheadAttention(embedding_dim, 4)
```

### 多尺度训练

```python
def multi_scale_training():
    for nodes in [20, 50, 100]:
        trainer.num_nodes = nodes
        trainer.train(num_epochs=50)
```

## 故障排除

### 常见问题

1. **CUDA内存不足**:
   ```python
   # 减少批次大小和POMO大小
   batch_size=16, pomo_size=20
   ```

2. **收敛慢**:
   ```python
   # 尝试纯邻接矩阵配置，或调整学习率
   network_type='none', lr=5e-4
   ```

3. **数值不稳定**:
   ```python
   # 添加梯度裁剪
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   ```

### 调试技巧

```python
# 在训练循环中添加调试信息
if epoch % 10 == 0:
    print(f"邻接矩阵范围: [{adj_matrix.min():.3f}, {adj_matrix.max():.3f}]")
    print(f"logits范围: [{logits.min():.3f}, {logits.max():.3f}]")
```

## 最佳实践建议

### 🎯 配置选择建议

1. **首次使用**: 从纯邻接矩阵决策开始
2. **效果不满意**: 尝试线性层增强
3. **复杂问题**: 考虑MLP增强
4. **计算资源有限**: 坚持使用纯邻接矩阵

### 📈 训练建议

1. **预热训练**: 先用较小问题规模训练
2. **检查点保存**: 定期保存模型检查点
3. **学习率调度**: 使用学习率衰减
4. **早停策略**: 监控验证性能，避免过拟合

## 参考文献

1. **POMO方法**: mtnco项目的VRP求解实现
2. **扩散模型**: difusco项目的TSP扩散模型  
3. **强化学习**: REINFORCE算法和优势估计
4. **邻接矩阵决策**: 本项目的核心创新

## 总结

本项目实现了一个**以扩散模型邻接矩阵为主导**的TSP求解方法，具有以下特点：

✅ **简单高效**: 主要依赖扩散模型输出，计算开销小  
✅ **灵活可配**: 支持多种复杂度配置，适应不同需求  
✅ **训练稳定**: 避免复杂网络结构，训练过程稳定  
✅ **效果良好**: 在保持简单性的同时获得竞争性的结果  

**推荐使用纯邻接矩阵决策或线性层增强配置**，在大多数情况下都能获得良好的效果。 