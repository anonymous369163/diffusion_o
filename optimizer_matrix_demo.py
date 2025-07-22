# %% 导入必要的库
from __future__ import annotations
import sys 
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch神经网络函数
import matplotlib.pyplot as plt  # 绘图库
import pandas as pd  # 数据处理库
from typing import Optional, Tuple, Dict, Any  # 类型提示 
import time


# 设置随机种子以确保结果可复现
torch.manual_seed(42)   
np.random.seed(42)

# 设置计算设备（优先使用GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OR_TOOLS_AVAILABLE =  True

# 输出当前运行环境信息
print(f"使用设备: {device}")
print(f"OR-Tools可用: {OR_TOOLS_AVAILABLE}")

# 修改: 添加difusco路径到sys.path
difusco_path = '/home/yuepeng/codes/difusco_cross_pro/difusco'
if difusco_path not in sys.path:    
    sys.path.append(difusco_path)
    print(f"已将 {difusco_path} 添加到 sys.path")

# 尝试从项目导入真实的函数
from difusco.pl_tsp_model import (
    calculate_euclidean_distance_batch,
    calculate_tour_cost_batch,
    calculate_tour_cost_batch_pomo,
    greedy_solver_batch_pomo
)
print("成功导入项目中的真实函数")

# %% 测试用的 TSPModel 类
## =====================================================================================================
class TestTSPModel:
    def __init__(self):
        # 强化学习相关参数 
        self.rl_baseline_decay = 0.95
        self.rl_baseline = None
        self.pomo_temperature = 0.01 # POMO温度参数 
        
        # 调试参数
        self.rl_debug = False
        self.add_prior = True  
    
    def _compute_ground_truth_costs(self, gt_tour, distance_matrices):
        """计算真实最优路径的成本
        注意这里的gt_tour没有返回到起点，这里加到回到节点的处理
        """
        gt_tours_list = []
        for b in range(gt_tour.shape[0]):
            gt_tour_b = gt_tour[b].cpu().numpy().tolist()
            gt_tour_b.append(gt_tour_b[0])  # 添加回到起点
            gt_tours_list.append(gt_tour_b)
        
        gt_costs = calculate_tour_cost_batch(gt_tours_list, distance_matrices)
        return gt_costs
    
    def _compute_rewards_and_advantages(self, pred_costs, gt_costs, batch_size):
        """计算奖励和优势函数"""
        # 扩展gt_costs以匹配POMO维度 
        num_starts = pred_costs.shape[0] // batch_size
        
        gt_costs_expanded = gt_costs.unsqueeze(1).expand(-1, num_starts).reshape(-1)
        
        # 计算奖励 (负的相对成本差异)
        cost_diff = pred_costs - gt_costs_expanded
        relative_cost_diff = cost_diff / (gt_costs_expanded + 1e-8)  # 避免除以0
        
        # 限制相对成本差异的范围以防止极端值
        # relative_cost_diff = torch.clamp(relative_cost_diff, min=-10.0, max=10.0)
        rewards = -relative_cost_diff  # 成本越低，奖励越高
        
        # 对于POMO，选择每个样本中最好的路径来计算基线
        rewards_reshaped = rewards.reshape(batch_size, -1)
        best_rewards = torch.max(rewards_reshaped, dim=1)[0]
        
        # 更新基线 (使用指数移动平均)
        current_baseline = best_rewards.mean().detach() 
        
        if self.rl_baseline is None:
            self.rl_baseline = current_baseline
        else:
            self.rl_baseline = self.rl_baseline_decay * self.rl_baseline + (1 - self.rl_baseline_decay) * current_baseline
        
        # 计算优势函数 (奖励减去基线)
        advantages = rewards - self.rl_baseline
        
        # 限制优势函数的范围
        # advantages = torch.clamp(advantages, min=-self.max_advantage, max=self.max_advantage)
        
        return rewards, advantages
    
    def _compute_rl_metrics(self, rl_loss, rewards, pred_costs, gt_costs, batch_size):
        """计算并返回强化学习相关指标""" 
        num_starts = pred_costs.shape[0] // batch_size
        
        # 重新计算相对成本差异用于指标
        gt_costs_expanded = gt_costs.unsqueeze(1).expand(-1, num_starts).reshape(-1)
        relative_cost_diff = (pred_costs - gt_costs_expanded) / (gt_costs_expanded + 1e-8)
        
        # 计算最佳成本
        pred_costs_reshaped = pred_costs.reshape(batch_size, num_starts)
        best_pred_costs = pred_costs_reshaped.min(dim=1)[0]
        
        # 计算各种指标
        best_rewards = rewards.reshape(batch_size, num_starts).max(dim=1)[0]
        
        metrics = {
            'rl_loss': rl_loss.item(),
            'avg_reward': rewards.mean().item(),
            'best_reward': best_rewards.mean().item(),
            'baseline': self.rl_baseline.item() if self.rl_baseline is not None else 0.0,
            'avg_pred_cost': pred_costs.mean().item(),
            'best_pred_cost': best_pred_costs.mean().item(),
            'avg_gt_cost': gt_costs.mean().item(),
            'cost_gap_percent': (relative_cost_diff * 100).mean().item(),
            'best_cost_gap_percent': ((best_pred_costs - gt_costs) / (gt_costs + 1e-8) * 100).mean().item(),
            'pomo_temperature': self.pomo_temperature
        }
        
        return metrics
    
    def compute_reinforcement_learning_loss(self, x0_pred, points, gt_tour, current_problem_type, batch_idx):
        """计算强化学习损失函数""" 
        # 计算softmax概率
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1) 

        # 取边存在的概率 (第二个类别)
        adj_prob_matrix = x0_pred_prob[:, :, :, 1]  # shape: (batch_size, num_nodes, num_nodes)
        
        # 第二步：计算距离矩阵
        distance_matrices = calculate_euclidean_distance_batch(points[:, :, :2])
        
        # 第三步：使用POMO求解器生成路径 
        pred_tours, log_probs = greedy_solver_batch_pomo(
                                                        adj_prob_matrix, 
                                                        temperature=self.pomo_temperature, 
                                                        points_with_features=points, 
                                                        problem_type=current_problem_type,
                                                        add_prior=self.add_prior, 
                                                        distance_matrices=distance_matrices, 
                                                        test_mode=False, 
                                                        debug=self.rl_debug
                                                        )
        
        # 第四步：计算预测路径成本
        pred_costs = calculate_tour_cost_batch_pomo(pred_tours, distance_matrices, problem_type=current_problem_type)
        
        # 第五步：计算真实最优路径成本
        gt_costs = self._compute_ground_truth_costs(gt_tour, distance_matrices)
        
        # 第六步：计算奖励和优势
        rewards, advantages = self._compute_rewards_and_advantages(
            pred_costs, gt_costs, points.shape[0]
        )
        
        # 第七步：计算REINFORCE损失
        rl_loss = -(log_probs * advantages.detach()).mean() 
        
        # 第八步：计算指标
        metrics = self._compute_rl_metrics(rl_loss, rewards, pred_costs, gt_costs, points.shape[0])
 
        return rl_loss, metrics, pred_tours, pred_costs 

# %% 从rl_debug.pkl导入调试数据
## =====================================================================================================
import pickle

print("正在从rl_debug.pkl加载调试数据...")
try:
    with open('rl_debug.pkl', 'rb') as f:
        debug_data = pickle.load(f)
    
    # 提取数据
    x0_pred = debug_data['x0_pred'] 
    points = debug_data['points']
    gt_tour = debug_data['gt_tour']
    problem_type = debug_data['current_problem_type']
    batch_idx = debug_data['batch_idx']
    adj_matrix = debug_data['adj_matrix']
    
    print("✅ 调试数据加载成功!")
    print(f"- x0_pred形状: {x0_pred.shape}")
    print(f"- points形状: {points.shape}")
    print(f"- gt_tour形状: {gt_tour.shape}")
    print(f"- 问题类型: {problem_type}")
    print(f"- batch_idx: {batch_idx}")
    
except Exception as e:
    print(f"❌ 调试数据加载失败: {e}")
    print("将使用生成的测试数据继续...")

# 取出第一个案例进行分析
batch_size = 1
batch_idx = 0
x0_pred = x0_pred[batch_idx:batch_idx+1]  # 保持维度    
points = points[batch_idx:batch_idx+1]
gt_tour = gt_tour[batch_idx:batch_idx+1]
adj_matrix = adj_matrix[batch_idx:batch_idx+1]

print(f"\n分析第 {batch_idx} 个案例:")
print(f"- x0_pred形状: {x0_pred.shape}")
print(f"- points形状: {points.shape}")
print(f"- gt_tour形状: {gt_tour.shape}")

# %%  将adj_matrix转换为x0_pred格式
adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()  # [1,51,51,2]
adj_matrix_onehot = adj_matrix_onehot.permute(0,3,1,2)  # [1,2,51,51]
print(adj_matrix_onehot.shape)  # 验证形状
x0_pred = adj_matrix_onehot
# %% 改进的约束感知优化更新方法  
## =====================================================================================================
def extract_adjacency_probabilities(x0_pred):
    """从x0_pred中提取邻接矩阵概率"""
    # x0_pred shape: (batch_size, 2, num_nodes, num_nodes) 
    x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
    adj_prob_matrix = x0_pred_prob[:, :, :, 1]  # 取边存在的概率
    return adj_prob_matrix

def constrained_optimizer_step(x0_pred, gradient, batch_idx=0,
                               depot_index=0,
                               use_local_search=True,
                               use_tensor_version=False,  # 新增：选择使用tensor版本还是numpy版本
                               alpha_cc=0.3,
                               alpha_depot=0.5,
                               # 局部搜索参数
                               local_search_moves=3,
                               local_transfer_ratio=0.3,
                               local_min_weight=0.02,
                               local_improvement_threshold=1e-5):
    """
    约束感知的优化器步骤 
    
    Args:
        x0_pred: torch.Tensor - 当前预测 (batch_size, 2, num_nodes, num_nodes)
        gradient: torch.Tensor - 梯度 (batch_size, 2, num_nodes, num_nodes) 
        batch_idx: int - 处理的批次索引
        depot_index: int - depot节点索引
        use_local_search: bool - 是否在KL_Sinkhorn基础上启用局部搜索
        use_tensor_version: bool - 是否使用tensor版本（支持GPU加速）还是numpy版本
        alpha_cc: float - 客户-客户步长参数
        alpha_depot: float - depot步长参数
        local_search_moves: int - 局部搜索最大移动次数
        local_transfer_ratio: float - 权重转移比例
        local_min_weight: float - 参与转移的最小权重阈值
        local_improvement_threshold: float - 局部改进阈值
        
    Returns:
        torch.Tensor - 更新后的x0_pred
        dict - 包含更新信息的字典
    """
    device = x0_pred.device 
    
    # 提取当前批次的数据
    x0_pred_single = x0_pred[batch_idx]  # (2, num_nodes, num_nodes)
    grad_single = gradient[batch_idx]    # (2, num_nodes, num_nodes)
    
    # 提取邻接矩阵概率 (只取边存在的概率)
    adj_prob = extract_adjacency_probabilities(x0_pred_single.unsqueeze(0))[0]  # (num_nodes, num_nodes)
    grad_adj = grad_single[1]  # 只处理边存在的梯度
    
    # 获取车辆数K
    K = adj_prob.shape[0] - 1  # depot=0, 其余为客户
    
    # 准备KL+Sinkhorn版本的参数
    kl_kwargs = {
        'depot': depot_index,
        'K': K,
        'eta_cc': 1.0,
        'alpha_cc': alpha_cc,
        'eta_depot': 1.0,
        'alpha_depot': alpha_depot,
        'sinkhorn_max_iter': 100,
        'forbid_self_loop': True,
        'force_return_to_depot': True
    }
    
    # 根据版本选择进行不同的处理
    if use_tensor_version:
        # 使用Tensor版本 (支持GPU加速)
        if use_local_search:
            # 使用带局部搜索的tensor版本
            kl_kwargs.update({
                'enable_local_search': True,
                'local_search_moves': local_search_moves,
                'local_transfer_ratio': local_transfer_ratio,
                'local_min_weight': local_min_weight,
                'local_improvement_threshold': local_improvement_threshold
            })
            C_new_tensor, info = gradient_assignment_vrp_KL_Sinkhorn_with_local_search_tensor(
                adj_prob, grad_adj, **kl_kwargs
            )
        else:
            # Tensor版本的原始KL+Sinkhorn (目前暂不支持，使用numpy版本)
            C = adj_prob.detach().cpu().numpy()
            g = grad_adj.detach().cpu().numpy()
            C_new, info = gradient_assignment_vrp_KL_Sinkhorn(C, g, **kl_kwargs)
            C_new_tensor = torch.from_numpy(C_new).float().to(device)
    else:
        # 使用Numpy版本 (原始版本)
        C = adj_prob.detach().cpu().numpy()
        g = grad_adj.detach().cpu().numpy()
        
        if use_local_search:
            # 使用带局部搜索的增强版本
            kl_kwargs.update({
                'enable_local_search': True,
                'local_search_moves': local_search_moves,
                'local_transfer_ratio': local_transfer_ratio,
                'local_min_weight': local_min_weight,
                'local_improvement_threshold': local_improvement_threshold
            })
            C_new, info = gradient_assignment_vrp_KL_Sinkhorn_with_local_search(
                C, g, **kl_kwargs
            )
        else:
            # 使用原始KL+Sinkhorn版本
            C_new, info = gradient_assignment_vrp_KL_Sinkhorn(
                C, g, **kl_kwargs
            )
        
        # 转换回PyTorch tensor
        C_new_tensor = torch.from_numpy(C_new).float().to(device)
    
    # 构造完整的更新
    x0_pred_updated = x0_pred.clone()
    
    # 更新边存在的logit
    adj_prob_updated = torch.clamp(C_new_tensor, min=1e-8, max=1.0-1e-8)
    
    # 转换回logit形式
    x0_pred_updated[batch_idx, 1] = torch.log(adj_prob_updated + 1e-8)
    x0_pred_updated[batch_idx, 0] = torch.log(1 - adj_prob_updated + 1e-8)
    
    # 返回更新后的张量和信息
    return x0_pred_updated, info

# %% 定义最终的优化器
## =====================================================================================================
from torch.optim.optimizer import Optimizer
class HybridOptimizer(Optimizer):
    """约束感知优化器：直接使用约束感知的更新方向"""
    
    def __init__(self, params, total_steps, lr=0.01, 
                 depot_index=0,
                 use_local_search=True,
                 use_tensor_version=True,  # 新增：选择使用tensor版本还是numpy版本
                 local_search_moves=5,
                 local_transfer_ratio=0.5,
                 local_min_weight=0.01,
                 local_improvement_threshold=5e-6,
                 **kwargs):
        # 设置默认参数
        defaults = dict(
            lr=lr, 
            depot_index=depot_index,
            use_local_search=use_local_search,
            use_tensor_version=use_tensor_version,
            local_search_moves=local_search_moves,
            local_transfer_ratio=local_transfer_ratio,
            local_min_weight=local_min_weight,
            local_improvement_threshold=local_improvement_threshold,
            **kwargs
        )
        super().__init__(params, defaults)
        
        # 初始化优化器状态
        self.step_count = 0
        self.total_steps = total_steps
        self.ema_gradient = torch.zeros_like(params[0].data)
        
        # 保存约束相关参数，方便访问
        self.constraint_kwargs = { 
            'depot_index': depot_index,
            'use_local_search': use_local_search,
            'use_tensor_version': use_tensor_version,
            'local_search_moves': local_search_moves,
            'local_transfer_ratio': local_transfer_ratio,
            'local_min_weight': local_min_weight,
            'local_improvement_threshold': local_improvement_threshold,
            **kwargs
        }
    
    
    def step(self):
        """执行一步优化"""
        self.step_count += 1 
        
        for group in self.param_groups: 
            depot_index = group['depot_index']
            
            for param in group['params']:
                if param.grad is not None:
                    with torch.no_grad():  
                        param.data, info = constrained_optimizer_step(
                            param, param.grad,
                            depot_index=depot_index,
                            **{k: v for k, v in group.items() if k not in [
                                'params', 'lr', 'depot_index',
                                'top_k_edges', 'gradient_threshold', 'cost_current', 'cost_best'
                            ]}
                        )
    
    def get_constraint_metrics(self):
        """获取约束相关指标"""
        metrics = {}
        
        # 如果有优化信息，返回最新的信息
        if hasattr(self, '_optimization_info') and self._optimization_info:
            latest_info = self._optimization_info[-1]
            metrics.update({
                'kl_divergence': latest_info.get('kl_divergence', 0.0), 
                'initial_quality': latest_info.get('initial_quality', 0.0),
                'current_alpha_cc': latest_info.get('current_alpha_cc', 0.0),
                'current_alpha_depot': latest_info.get('current_alpha_depot', 0.0)
            })
            
        return metrics
    
    @property
    def lr(self):
        """获取学习率（从第一个参数组）"""
        return self.param_groups[0]['lr']
    
    @property
    def constraint_frequency(self):
        """约束更新频率（默认每步都更新）"""
        return 1    


# %%  可视化函数 （辅助函数生成热力图和路径图）
## =====================================================================================================
def visualize_optimization_step(step, points, adj_prob, best_tours_current=None, 
                                current_cost=None, gt_cost=None, gt_tour=None,
                                cost_history=None, best_cost_history=None, 
                                step_history=None, fig_size=(16, 8)):
    """Simplified visualization function - show key plots: adjacency matrix heatmap, optimal solution heatmap, generated path and cost trend"""
    
    # Use 2x2 layout to show four core visualizations
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    
    coordinates = points[0, :, :2].detach().cpu().numpy()
    num_nodes = len(coordinates)
    
    # 1. Current predicted adjacency matrix heatmap (top left)
    adj_prob_np = adj_prob.detach().cpu().numpy()
    im1 = axes[0,0].imshow(adj_prob_np, cmap='YlOrRd', vmin=0, vmax=1)
    axes[0,0].set_title(f'Step {step}: Current Prediction Heatmap')
    axes[0,0].set_xlabel('To Node')
    axes[0,0].set_ylabel('From Node')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 2. Optimal solution heatmap (Ground Truth)
    optimal_heatmap = np.zeros((num_nodes, num_nodes))
    tour_to_use = None
    heatmap_title = "Optimal Solution Heatmap (Unknown)"
    
    # Prioritize using Ground Truth path
    if gt_tour is not None:
        try:
            tour_to_use = gt_tour[0].detach().cpu().numpy()
            heatmap_title = "Optimal Solution Heatmap (Ground Truth)"
        except:
            pass
    
    # If no GT path, use current best path
    if tour_to_use is None and best_tours_current is not None:
        try:
            tour_to_use = best_tours_current[0].detach().cpu().numpy()
            heatmap_title = "Current Best Solution Heatmap"
        except:
            pass
    
    # Generate heatmap
    if tour_to_use is not None:
        try:
            valid_path = [node for node in tour_to_use if 0 <= node < num_nodes]
            
            # Create heatmap based on path
            if len(valid_path) > 1:
                for i in range(len(valid_path) - 1):
                    from_node = valid_path[i]
                    to_node = valid_path[i + 1]
                    optimal_heatmap[from_node, to_node] = 1.0
                
                # For TSP/VRP, return to start point (depot)
                if len(valid_path) >= 2:
                    optimal_heatmap[valid_path[-1], 0] = 1.0
        except Exception as e:
            print(f"Error creating optimal solution heatmap: {e}")
            heatmap_title = "Optimal Solution Heatmap (Error)"
    
    im2 = axes[0,1].imshow(optimal_heatmap, cmap='Blues', vmin=0, vmax=1)
    axes[0,1].set_title(f'Step {step}: {heatmap_title}')
    axes[0,1].set_xlabel('To Node')
    axes[0,1].set_ylabel('From Node')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 3. Generated path plot (bottom left)
    axes[1,0].scatter(coordinates[:, 0], coordinates[:, 1], s=200, c='red', alpha=0.7, zorder=5)
    axes[1,0].scatter(coordinates[0, 0], coordinates[0, 1], s=300, c='blue', marker='s', alpha=0.9, zorder=6, label='Depot')
    
    # Add node labels
    for i in range(len(coordinates)):
        axes[1,0].annotate(f'{i}', (coordinates[i, 0], coordinates[i, 1]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Draw generated path
    if best_tours_current is not None:
        try:
            # Get generated path from first batch
            current_path = best_tours_current[0].detach().cpu().numpy()
            valid_path = [node for node in current_path if 0 <= node < num_nodes]
            if len(valid_path) > 1:
                for i in range(len(valid_path) - 1):
                    start = coordinates[valid_path[i]]
                    end = coordinates[valid_path[i + 1]]
                    axes[1,0].plot([start[0], end[0]], [start[1], end[1]], 
                                 'blue', linewidth=3, alpha=0.8)
                # Draw path back to depot
                if len(valid_path) >= 2:
                    start = coordinates[valid_path[-1]]
                    end = coordinates[0]  # back to depot
                    axes[1,0].plot([start[0], end[0]], [start[1], end[1]], 
                                 'blue', linewidth=3, alpha=0.8, linestyle='--', label='Return to Depot')
                axes[1,0].set_title(f'Step {step}: Generated Path')
        except:
            axes[1,0].set_title(f'Step {step}: Path Generation Error')
    else:
        # Draw high probability connections as alternatives
        threshold = 0.3
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                if adj_prob_np[i,j] > threshold:
                    alpha = float(adj_prob_np[i,j])
                    linewidth = 1 + 2 * alpha
                    axes[1,0].plot([coordinates[i,0], coordinates[j,0]], 
                                 [coordinates[i,1], coordinates[j,1]], 
                                 'g-', alpha=alpha, linewidth=linewidth)
        axes[1,0].set_title(f'Step {step}: Probability Connections')
    
    axes[1,0].set_xlim(-0.1, 1.1)
    axes[1,0].set_ylim(-0.1, 1.1)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # 4. Cost trend plot
    if cost_history is not None and len(cost_history) > 0:
        # Ensure valid step_history
        if step_history is None or len(step_history) != len(cost_history):
            step_history = list(range(len(cost_history)))
        
        axes[1,1].plot(step_history, cost_history, 'b-', linewidth=2, alpha=0.8, label='Current Cost')
        
        if best_cost_history is not None and len(best_cost_history) == len(cost_history):
            axes[1,1].plot(step_history, best_cost_history, 'r-', linewidth=2, alpha=0.8, label='Best Cost')
        
        # Add current cost and GT cost info
        if current_cost is not None:
            axes[1,1].axhline(y=current_cost, color='blue', linestyle='--', alpha=0.7, label=f'Current: {current_cost:.3f}')
        
        if gt_cost is not None:
            axes[1,1].axhline(y=gt_cost, color='orange', linestyle='--', alpha=0.7, label=f'GT: {gt_cost:.3f}')
        
        axes[1,1].set_title('Cost Optimization Trend')
        axes[1,1].set_xlabel('Optimization Steps')
        axes[1,1].set_ylabel('Cost')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
    else:
        axes[1,1].text(0.5, 0.5, 'No cost history\navailable', 
                      horizontalalignment='center', verticalalignment='center', 
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('Cost Trend: N/A')
    
    plt.tight_layout()
    plt.show()
    plt.close()
# %% 随机初始化x0_pred

# 随机初始化x0_pred
print("\n" + "="*80)
print("随机初始化x0_pred")
print("="*80)

# 获取维度信息
batch_size = points.shape[0]
num_nodes = points.shape[1]

# 随机初始化x0_pred
x0_pred = torch.randn(batch_size, 2, num_nodes, num_nodes) * 0.01

# 将对角线元素设为很小的值,避免自环
diag_mask = torch.eye(num_nodes).unsqueeze(0).unsqueeze(0).expand(batch_size, 2, -1, -1)
x0_pred = x0_pred.masked_fill(diag_mask == 1, -10.0)

print(f"x0_pred形状: {x0_pred.shape}")
print(f"x0_pred初始化范围: [{x0_pred.min():.4f}, {x0_pred.max():.4f}]")

# %% gradient_assignment_vrp_KL_Sinkhorn
# ================================================================

def _row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = mat.copy()
    rs = out.sum(axis=1, keepdims=True)
    rs = np.maximum(rs, eps)
    out /= rs
    return out

def sinkhorn_doubly_stochastic(
    scores: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-9,
    eps: float = 1e-12,
    anchor_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对非负矩阵 `scores` 做 Sinkhorn 迭代，得到近似双随机矩阵。

    参数
    ----
    scores : (n,n) 非负矩阵；未归一化权重。
    mask   : (n,n) bool，可行边 True，不可行 False；不可行位置强制 eps。
    max_iter : 最大迭代次数。
    tol : 收敛阈值（行/列偏差 L1）。
    eps : 数值最小值，避免除零。
    anchor_scale : (n,) 可选，每行一个放大倍数；用于"锚定强边"时加权。
        具体做法：若 anchor_scale[i] > 1，则该行的最大元素乘以该系数。
        （简单稳健，不严格冻结；避免列冲突。）

    返回
    ----
    X : (n,n) 双随机矩阵（行列和 ≈1）。
    """
    M = np.array(scores, dtype=np.float64, copy=True)
    n = M.shape[0]
    if mask is not None:
        M = np.where(mask, M, eps)

    # 锚定：对每行 top1 乘 anchor scale
    if anchor_scale is not None:
        assert anchor_scale.shape[0] == n
        for i in range(n):
            if anchor_scale[i] > 1.0:
                j = np.argmax(M[i])
                M[i, j] *= anchor_scale[i]

    # 初始化 u,v
    u = np.ones(n, dtype=np.float64) / n
    v = np.ones(n, dtype=np.float64) / n

    # Sinkhorn 迭代
    # 形式：X = diag(u) @ M @ diag(v)
    # 每次标准化行/列使其和=1
    for _ in range(max_iter):
        # 行标准化
        row_sums = M.sum(axis=1)
        row_sums = np.maximum(row_sums, eps)
        M = M / row_sums[:, None]
        # 列标准化
        col_sums = M.sum(axis=0)
        col_sums = np.maximum(col_sums, eps)
        M = M / col_sums[None, :]
        # 收敛检测（行列和偏差）
        if np.max(np.abs(row_sums - 1.0)) < tol and np.max(np.abs(col_sums - 1.0)) < tol:
            break

    # 最终归一化（防护）
    M = _row_normalize(M, eps=eps)
    col_sums = M.sum(axis=0, keepdims=True)
    M /= np.maximum(col_sums, eps)
    return M


def row_normalize(mat, mask=None, eps=1e-12):
    M = mat.copy()
    if mask is not None:
        M = np.where(mask, M, 0.0)
    rowsum = M.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, eps, None)
    M = M / rowsum
    return M
# ------------------------------
# depot行 soft top-k 截断/稀疏化
# ------------------------------
def soft_topk_normalize(row, k, temperature=1.0, hard=False, eps=1e-12):
    """
    将一行概率分布限制为最多K个显著元素（soft）。
    row: shape (M,) ; 不含 depot 列（一般 = 客户子集）
    k: 最大车辆数
    temperature: softmax温度（越小越尖锐）
    hard: True 则硬 top-k (其余置eps), False 则使用温度放大 top-k 与缩小其余
    """
    M = row.shape[0]
    if k >= M:
        # 无需截断
        r = row.copy()
        s = r.sum()
        return r / (s if s > eps else M)

    # 找 top-k 索引
    idx = np.argpartition(-row, k-1)[:k]   # 未排序
    topk_mask = np.zeros(M, dtype=bool)
    topk_mask[idx] = True

    r = np.zeros_like(row)

    if hard:
        # 仅保留 top-k，其余置 eps
        r[topk_mask] = row[topk_mask]
        r[~topk_mask] = eps
    else:
        # soft: top-k 放大，非topk 缩小
        scale_large = 1.0
        scale_small = np.exp(-5.0 / temperature)  # 可调；温度越小 -> 缩得越狠
        r[topk_mask] = row[topk_mask] * scale_large
        r[~topk_mask] = np.maximum(row[~topk_mask] * scale_small, eps)

    # 归一化
    s = r.sum()
    if s <= eps:
        r = np.ones(M) / M
    else:
        r = r / s
    return r

# ------------------------------
# 核心：VRP KL+Sinkhorn 梯度指派步
# ------------------------------
def gradient_assignment_vrp_KL_Sinkhorn(
    C, G,
    depot=0,
    K=2,
    eta_cc=1.0,       # 客户-客户梯度温度
    alpha_cc=0.5,     # 客户-客户注入比例
    eta_depot=1.0,    # depot->客户 梯度温度
    alpha_depot=0.7,  # depot行注入比例
    sinkhorn_max_iter=100,
    forbid_self_loop=True,
    force_return_to_depot=True,
    min_return_mass=1e-3,
    depot_topk_hard=False,
    depot_topk_temperature=1.0,
    eps=1e-12
):
    """
    基于 KL + Sinkhorn 的 VRP 梯度指派更新（soft 结构）
    - 在客户子块 (1..N-1,1..N-1) 做 KL+Sinkhorn -> 排列结构
    - depot 行 soft top-k 控制车辆数 ≤ K
    - 客户行剩余质量自动分配给 depot 列，实现回 depot（soft）
    """
    C = np.asarray(C, dtype=float)
    G = np.asarray(G, dtype=float)
    N = C.shape[0]
    assert C.shape == (N, N)
    assert G.shape == (N, N)
    assert depot >= 0 and depot < N

    clients = [i for i in range(N) if i != depot]

    # ---- Step 1: 客户-客户 KL + exp 梯度步
    Cc = C[np.ix_(clients, clients)]
    Gc = G[np.ix_(clients, clients)]
    Sc = Cc * np.exp(-eta_cc * Gc)
    Sc = np.maximum(Sc, eps)

    # mask 禁自环
    if forbid_self_loop:
        mask_c = np.ones_like(Sc, dtype=bool)
        np.fill_diagonal(mask_c, False)
        Sc = np.where(mask_c, Sc, 0.0)
    else:
        mask_c = None

    # Sinkhorn -> 双随机近似
    Xc = sinkhorn_doubly_stochastic(
        Sc, mask=mask_c, max_iter=sinkhorn_max_iter, tol=1e-9, eps=eps
    )

    # ---- Step 2: 混合客户子块回全矩阵
    X_full = C.copy()
    X_full[np.ix_(clients, clients)] = Xc
    C_mid = (1 - alpha_cc) * C + alpha_cc * X_full

    # ---- Step 3: depot->客户 soft top-k（受梯度引导）
    depot_row = C_mid[depot, clients]
    depot_grad = G[depot, clients]
    # KL 指数步
    depot_step = depot_row * np.exp(-eta_depot * depot_grad)
    depot_step = np.maximum(depot_step, eps)
    depot_step = depot_step / depot_step.sum()

    # soft top-k 限制车辆数
    depot_step = soft_topk_normalize(
        depot_step, k=K, temperature=depot_topk_temperature, hard=depot_topk_hard, eps=eps
    )

    # 注入回 depot 行
    depot_new_row = (1 - alpha_depot) * depot_row + alpha_depot * depot_step

    # 合成 depot 行（保持自身对角=0）
    C_mid[depot, clients] = depot_new_row
    C_mid[depot, depot] = 0.0

    # ---- Step 4: 客户行强制"剩余质量 = 回 depot"
    if force_return_to_depot:
        for i in clients:
            # 当前行
            row_client = C_mid[i, :]
            # 去 depot + 去客户
            to_clients = row_client[clients].sum()
            # 目标行和=1: 让 C[i,depot] = 1 - to_clients（若负则截断）
            ret = 1.0 - to_clients
            if ret < min_return_mass:
                ret = min_return_mass  # 防止完全不回 depot
                # 再缩放客户子块
                scale = (1.0 - ret) / max(to_clients, eps)
                C_mid[i, clients] = row_client[clients] * scale
            C_mid[i, depot] = ret

        # 行再归一
        C_mid = row_normalize(C_mid, mask=None, eps=eps)
    else:
        # 简单行归一
        C_mid = row_normalize(C_mid, eps=eps)

    # ---- Step 5: 数值清理 & depot 行归一（防止浮点偏差）
    C_mid[depot, depot] = 0.0
    # C_mid = row_normalize(C_mid, eps=eps)

    return C_mid, {
        "Xc_client_block": Xc,
        "depot_row_new": depot_new_row,
        "num_active_starts_soft": float((depot_new_row > 1e-6).sum()),
    }

# %%  gradient_assignment_vrp_KL_Sinkhorn_with_local_search
""" 
混合增强版VRP梯度指派 - 平衡版本说明
=====================================

经过调整，gradient_assignment_vrp_hybrid_enhanced 现在具有更好的更新能力： 

内部机制优化：
1. 自适应调整更温和：质量因子最大为2.0（而非3.0）
2. 动量权重提升至0.2（而非0.1）：更好利用历史信息  
3. 局部搜索阈值降低：更容易触发局部改进
4. 边权重转移量增加：从0.02提升到0.05
5. 回退条件大幅放宽：质量下降70%或变化超过60%才回退

在HybridOptimizer中启用：set use_hybrid=True, use_enhanced=False
"""

# 添加基于KL_Sinkhorn的局部搜索增强版本
def gradient_assignment_vrp_KL_Sinkhorn_with_local_search(
    C, G,
    depot=0,
    K=2,
    eta_cc=1.0,       # 客户-客户梯度温度
    alpha_cc=0.5,     # 客户-客户注入比例
    eta_depot=1.0,    # depot->客户 梯度温度
    alpha_depot=0.7,  # depot行注入比例
    sinkhorn_max_iter=100,
    forbid_self_loop=True,
    force_return_to_depot=True,
    min_return_mass=1e-3,
    depot_topk_hard=False,
    depot_topk_temperature=1.0,
    eps=1e-12,
    # 新增局部搜索参数
    enable_local_search=True,    # 是否启用局部搜索
    local_search_moves=3,         # 局部搜索的最大移动次数
    local_transfer_ratio=0.3,     # 局部搜索中权重转移的比例
    local_min_weight=0.02,        # 局部搜索中参与转移的最小权重阈值
    local_improvement_threshold=1e-5,  # 局部改进的阈值
):
    """
    基于 KL + Sinkhorn 的 VRP 梯度指派更新，带可选局部搜索增强
    
    基于原始 gradient_assignment_vrp_KL_Sinkhorn 函数，保持其优秀的核心性能，
    同时添加可选的局部搜索功能来进行微调优化。
    
    参数:
    - enable_local_search: 是否启用局部搜索（默认False保持原始性能）
    - local_search_moves: 局部搜索的最大移动次数
    - local_transfer_ratio: 权重转移比例
    - local_min_weight: 参与转移的最小权重阈值
    - local_improvement_threshold: 改进阈值
    """
    C = np.asarray(C, dtype=float)
    G = np.asarray(G, dtype=float)
    N = C.shape[0]
    assert C.shape == (N, N)
    assert G.shape == (N, N)
    assert depot >= 0 and depot < N

    clients = [i for i in range(N) if i != depot]

    # ---- Step 1: 客户-客户 KL + exp 梯度步 （保持原始逻辑）
    Cc = C[np.ix_(clients, clients)]
    Gc = G[np.ix_(clients, clients)]
    Sc = Cc * np.exp(-eta_cc * Gc)
    Sc = np.maximum(Sc, eps)

    # mask 禁自环
    if forbid_self_loop:
        mask_c = np.ones_like(Sc, dtype=bool)
        np.fill_diagonal(mask_c, False)
        Sc = np.where(mask_c, Sc, 0.0)
    else:
        mask_c = None

    # Sinkhorn -> 双随机近似
    Xc = sinkhorn_doubly_stochastic(
        Sc, mask=mask_c, max_iter=sinkhorn_max_iter, tol=1e-9, eps=eps
    )

    # ---- Step 2: 混合客户子块回全矩阵 （保持原始逻辑）
    X_full = C.copy()
    X_full[np.ix_(clients, clients)] = Xc
    C_mid = (1 - alpha_cc) * C + alpha_cc * X_full

    # ---- Step 3: depot->客户 soft top-k（受梯度引导） （保持原始逻辑）
    depot_row = C_mid[depot, clients]
    depot_grad = G[depot, clients]
    # KL 指数步
    depot_step = depot_row * np.exp(-eta_depot * depot_grad)
    depot_step = np.maximum(depot_step, eps)
    depot_step = depot_step / depot_step.sum()

    # soft top-k 限制车辆数
    depot_step = soft_topk_normalize(
        depot_step, k=K, temperature=depot_topk_temperature, hard=depot_topk_hard, eps=eps
    )

    # 注入回 depot 行
    depot_new_row = (1 - alpha_depot) * depot_row + alpha_depot * depot_step

    # 合成 depot 行（保持自身对角=0）
    C_mid[depot, clients] = depot_new_row
    C_mid[depot, depot] = 0.0

    # ---- Step 4: 客户行强制"剩余质量 = 回 depot" （保持原始逻辑）
    if force_return_to_depot:
        for i in clients:
            # 当前行
            row_client = C_mid[i, :]
            # 去 depot + 去客户
            to_clients = row_client[clients].sum()
            # 目标行和=1: 让 C[i,depot] = 1 - to_clients（若负则截断）
            ret = 1.0 - to_clients
            if ret < min_return_mass:
                ret = min_return_mass  # 防止完全不回 depot
                # 再缩放客户子块
                scale = (1.0 - ret) / max(to_clients, eps)
                C_mid[i, clients] = row_client[clients] * scale
            C_mid[i, depot] = ret

        # 行再归一
        C_mid = row_normalize(C_mid, mask=None, eps=eps)
    else:
        # 简单行归一
        C_mid = row_normalize(C_mid, eps=eps)

    # ---- Step 5: 新增局部搜索阶段 ----
    local_improvements = 0
    if enable_local_search and len(clients) > 1:
        def local_refinement(matrix, grad_matrix):
            """局部搜索微调：基于梯度信息进行边权重调整"""
            C_local = matrix.copy()
            improvements = 0
            
            for move in range(local_search_moves):
                best_improvement = 0
                best_move = None
                
                # 寻找最有潜力的调整
                for i in clients:
                    row = C_local[i, clients]  # 只考虑客户间的边
                    grad_row = grad_matrix[i, clients]
                    
                    # 找出权重较高的边和梯度最负的边
                    high_weight_idx = np.argsort(row)[-min(3, len(clients)):]  # 取最高的几个
                    low_grad_idx = np.argsort(grad_row)[:min(3, len(clients))]   # 取梯度最负的几个
                    
                    for hw_idx in high_weight_idx:
                        for lg_idx in low_grad_idx:
                            if hw_idx != lg_idx and row[hw_idx] > local_min_weight:
                                # 计算潜在改进（基于梯度差异）
                                potential = grad_row[hw_idx] - grad_row[lg_idx]
                                if potential > best_improvement:
                                    best_improvement = potential
                                    best_move = (i, clients[hw_idx], clients[lg_idx])
                
                # 如果找到改进，应用它
                if best_move and best_improvement > local_improvement_threshold:
                    i, j_from, j_to = best_move
                    transfer = min(local_transfer_ratio * C_local[i, j_from], 
                                 C_local[i, j_from] * 0.5)  # 限制转移量
                    C_local[i, j_from] -= transfer
                    C_local[i, j_to] += transfer
                    improvements += 1
                else:
                    break  # 没有找到改进，停止
            
            return C_local, improvements
        
        # 应用局部搜索（只对客户部分）
        C_refined, local_improvements = local_refinement(C_mid, G)
        
        if local_improvements > 0:
            def apply_stability_constraint(C_old, C_new, max_change_ratio=0.1):
                change = np.abs(C_new - C_old)
                max_allowed_change = max_change_ratio * np.abs(C_old + 1e-8)
                
                # 限制过大的变化
                excessive_change = change > max_allowed_change
                C_constrained = np.where(excessive_change, 
                                    C_old + np.sign(C_new - C_old) * max_allowed_change,
                                    C_new)
                return C_constrained
            # 保存原始C_mid用于稳定性约束
            C_original = C_mid.copy()
            
            # 保持depot相关的约束不变
            C_mid = C_refined.copy()
            
            # 应用稳定性约束，防止过大变化
            C_mid = apply_stability_constraint(C_original, C_mid, max_change_ratio=0.1)
            
            # 重新应用depot约束
            if force_return_to_depot:
                for i in clients:
                    row_client = C_mid[i, :]
                    to_clients = row_client[clients].sum()
                    ret = max(1.0 - to_clients, min_return_mass)
                    
                    if ret > min_return_mass and to_clients > eps:
                        scale = (1.0 - ret) / to_clients
                        C_mid[i, clients] = row_client[clients] * scale
                    C_mid[i, depot] = ret
            
            # 重新归一化
            C_mid = row_normalize(C_mid, mask=None, eps=eps)

    # ---- Step 6: 数值清理 & depot 行归一（防止浮点偏差）（保持原始逻辑）
    C_mid[depot, depot] = 0.0

    return C_mid, {
        "Xc_client_block": Xc,
        "depot_row_new": depot_new_row,
        "num_active_starts_soft": float((depot_new_row > 1e-6).sum()),
        "local_search_enabled": enable_local_search,
        "local_improvements": local_improvements,
        "method": "KL_Sinkhorn_with_local_search"
    } 

# %% 优化器对连接矩阵进行修正 （main 函数） 
## =====================================================================================================
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

model = TestTSPModel() 

# 超参数设置
learning_rate = 0.03 
top_k_edges = 2       # 每行考虑的候选边数量
gradient_threshold = 0.1  # 梯度阈值
num_iters = 500
gradient_accumulation_steps = 1  # 梯度累积步数
model.pomo_temperature = 0.01

# 重新初始化参数
# 将模型和输入数据放在同一设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
x0_pred = x0_pred.to(device)
points = points.to(device)
gt_tour = gt_tour.to(device)
print(f"模型和数据已移至设备: {device}")

x0_pred_optim = torch.nn.Parameter(x0_pred.clone().detach())
 
# 设置混合优化器
# 1. Adam优化器用于每步更新
adam_optimizer = torch.optim.Adam([x0_pred_optim], lr=learning_rate)

# 2. HybridOptimizer用于约束感知更新
hybrid_optimizer = HybridOptimizer(
    params=[x0_pred_optim],
    total_steps=num_iters,
    lr=learning_rate, 
    top_k_edges=top_k_edges,
    gradient_threshold=gradient_threshold
)

# 3. 梯度累积器
accumulated_gradients = None
gradient_count = 0

print(f"使用混合优化策略:")
print(f"  - Adam优化器学习率: {learning_rate}")
print(f"  - 约束感知优化器:")
print(f"    - 梯度累积步数: {gradient_accumulation_steps}") 
print(f"    - 每行考虑的候选边数量: {top_k_edges}")
print(f"    - 梯度阈值: {gradient_threshold}")

# 训练设置
log = defaultdict(list)

print(f"\n开始 RL 微调（{num_iters} 步）")

# 训练循环
visualization_steps = list(range(0, num_iters, 5))  # 每5步可视化一次

# 初始化性能趋势跟踪
cost_history = []
best_cost_history = []
gt_cost_ref = None
gap_history = []
step_history = []

pbar = tqdm(range(num_iters), desc="RL微调")
for step in pbar:
    # === 1. 计算损失和梯度 ===
    # adam_optimizer.zero_grad()  # 清除Adam优化器的梯度
    hybrid_optimizer.zero_grad()
    
    # 计算损失
    rl_loss, metrics, pred_tours, pred_costs = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred_optim,
        points=points,
        gt_tour=gt_tour,
        current_problem_type=problem_type,
        batch_idx=batch_idx
    )
    
    # 提取当前成本和最佳成本用于更新优化器
    pred_costs_reshaped = pred_costs.reshape(batch_size, -1)
    current_best_cost = pred_costs_reshaped[0, pred_costs_reshaped.argmin(dim=1)[0]].item()
    
    # 跟踪历史最佳成本
    if step == 0:
        # 第一步初始化最佳成本
        historical_best_cost = current_best_cost
        gt_cost_ref = metrics['avg_gt_cost']  # 保存参考GT成本
    else:
        # 更新历史最佳成本
        historical_best_cost = min(historical_best_cost, current_best_cost)
    
    # 记录性能趋势数据
    cost_history.append(current_best_cost)
    best_cost_history.append(historical_best_cost)
    step_history.append(step)
    
    # 计算与GT的差距百分比
    if gt_cost_ref is not None and gt_cost_ref > 0:
        gap_percentage = (current_best_cost - gt_cost_ref) / gt_cost_ref * 100
        gap_history.append(gap_percentage)
    else:
        gap_history.append(0.0) 
    
    # 反向传播
    rl_loss.backward()
    
    # === 3. 每步使用Adam更新 ===
    # adam_optimizer.step()
    
    # === 4. 每N步使用HybridOptimizer处理累积梯度（使用指数平均） ===
    constraint_update_flag = False
    ema_decay = 0.95  # 指数平均衰减系数 
    # 初始化EMA梯度
    if not hasattr(hybrid_optimizer, 'ema_gradient'):
        hybrid_optimizer.ema_gradient = x0_pred_optim.grad.clone() if x0_pred_optim.grad is not None else torch.zeros_like(x0_pred_optim.data)
    else:
        hybrid_optimizer.ema_gradient = ema_decay * hybrid_optimizer.ema_gradient + (1 - ema_decay) * x0_pred_optim.grad if x0_pred_optim.grad is not None else torch.zeros_like(x0_pred_optim.data)

    avg_accumulated_gradient = hybrid_optimizer.ema_gradient / (1 - ema_decay ** (step + 1))  # 偏差修正

    # 临时保存当前梯度
    current_gradient = x0_pred_optim.grad.clone() if x0_pred_optim.grad is not None else None

    # 使用EMA梯度进行约束感知更新
    with torch.no_grad():
        x0_pred_optim.grad = avg_accumulated_gradient
        hybrid_optimizer.step()
        x0_pred_optim.grad = current_gradient

    constraint_update_flag = True
    
    # === 5. 记录日志 ===
    log["step"].append(step)
    log["loss"].append(rl_loss.item())
    log["constraint_update"].append(constraint_update_flag)  # 每步都记录是否进行约束更新
    for k, v in metrics.items():
        log[k].append(v)
    
    # 更新进度条
    progress_info = {
        'loss': f'{rl_loss.item():.4f}',
        'best_cost': f'{historical_best_cost:.4f}',
        'gap': f'{gap_history[-1]:.2f}%' if gap_history else '0.0%'
    }
    pbar.set_postfix(progress_info)

    # === 6. 可视化 ===
    if step in visualization_steps:
        with torch.no_grad():
            adj_prob = extract_adjacency_probabilities(x0_pred_optim)[0] # 获取当前邻接矩阵
        
        # 从pred_tours中筛选出最优路径 
        pred_costs_reshaped = pred_costs.reshape(batch_size, -1)
        pred_tours_reshaped = pred_tours.reshape(batch_size, -1, pred_tours.shape[-1]) 
        
        # 获取每个问题的最优路径索引
        best_path_indices = pred_costs_reshaped.argmin(dim=1)
        
        # 提取最优路径
        best_tours_current = torch.stack([
            pred_tours_reshaped[b, best_path_indices[b]] 
            for b in range(batch_size)
        ])
        # 在最优路径前加入0(depot)作为起点
        best_tours_current = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.long, device=best_tours_current.device),
            best_tours_current
        ], dim=1)
        
        # 提取当前最优路径的成本
        current_best_cost = pred_costs_reshaped[0, best_path_indices[0]].item()  # 取第一个批次的最优成本
        
        # 提取Ground Truth成本
        gt_cost = metrics['avg_gt_cost']

        visualize_optimization_step(step, points, adj_prob, best_tours_current=best_tours_current, 
                                   current_cost=current_best_cost, gt_cost=gt_cost, gt_tour=gt_tour,
                                   cost_history=cost_history, best_cost_history=best_cost_history,
                                   step_history=step_history)


print(f"\n微调完成！")
print(f"总约束更新次数: {len(log.get('constraint_update', []))}")

# 保存结果
x0_pred_final = x0_pred_optim.detach()
df_log = pd.DataFrame(log)

# %%  绘制训练过程曲线  
## =====================================================================================================
import matplotlib.pyplot as plt

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 绘制损失曲线
ax1.plot(df_log['step'], df_log['loss'], 'b-', label='RL Loss') 
ax1.set_title('Loss During Training')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss Value')
ax1.grid(True)
ax1.legend()

# 绘制成本曲线
ax2.plot(df_log['step'], df_log['avg_pred_cost'], 'r-', label='Avg Pred Cost')
ax2.plot(df_log['step'], df_log['best_pred_cost'], 'g-', label='Best Pred Cost')
ax2.axhline(y=df_log['avg_gt_cost'].mean(), color='b', linestyle='--', label='Optimal Cost')
ax2.set_title('Path Length During Training')
ax2.set_xlabel('Iteration') 
ax2.set_ylabel('Path Length')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# %% 评估基线 vs. 微调后性能
## =====================================================================================================
model.pomo_temperature = 0.0

# 基线评估
with torch.no_grad():
    base_loss, base_metrics, _, _ = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred.detach(),
        points=points,
        gt_tour=gt_tour,
        current_problem_type=problem_type,
        batch_idx=batch_idx
    )

# 微调后评估
with torch.no_grad():
    tuned_loss, tuned_metrics, _, _ = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred_optim.detach(),
        points=points,
        gt_tour=gt_tour,
        current_problem_type=problem_type,
        batch_idx=batch_idx
    )

# 计算成本差距
base_cost = base_metrics['avg_pred_cost']
tuned_cost = tuned_metrics['avg_pred_cost']
gt_cost = base_metrics['avg_gt_cost']

base_gap = (base_cost - gt_cost) / gt_cost * 100
tuned_gap = (tuned_cost - gt_cost) / gt_cost * 100

print(f"\n成本差距分析:")
print(f"基线成本: {base_cost:.4f}")
print(f"微调成本: {tuned_cost:.4f}")
print(f"最优成本: {gt_cost:.4f}")
print(f"基线 Gap: {base_gap:.2f}%")
print(f"微调 Gap: {tuned_gap:.2f}%")
print(f"Gap改进: {base_gap - tuned_gap:.2f}%")

# %% Tensor版本的辅助函数
## =====================================================================================================
import torch 

def _row_normalize_tensor(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Tensor版本的行归一化函数"""
    out = mat.clone()
    rs = out.sum(dim=1, keepdim=True)
    rs = torch.clamp(rs, min=eps)
    out = out / rs
    return out

def sinkhorn_doubly_stochastic_tensor(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_iter: int = 100,
    tol: float = 1e-9,
    eps: float = 1e-12,
    anchor_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """对非负矩阵 `scores` 做 Sinkhorn 迭代，得到近似双随机矩阵 (Tensor版本)。

    参数
    ----
    scores : (n,n) 非负tensor；未归一化权重。
    mask   : (n,n) bool tensor，可行边 True，不可行 False；不可行位置强制 eps。
    max_iter : 最大迭代次数。
    tol : 收敛阈值（行/列偏差 L1）。
    eps : 数值最小值，避免除零。
    anchor_scale : (n,) 可选tensor，每行一个放大倍数；用于"锚定强边"时加权。

    返回
    ----
    X : (n,n) 双随机tensor（行列和 ≈1）。
    """
    M = scores.clone().to(torch.float64)
    n = M.shape[0]
    device = M.device
    
    if mask is not None:
        M = torch.where(mask, M, eps)

    # 锚定：对每行 top1 乘 anchor scale
    if anchor_scale is not None:
        assert anchor_scale.shape[0] == n
        for i in range(n):
            if anchor_scale[i] > 1.0:
                j = torch.argmax(M[i])
                M[i, j] *= anchor_scale[i]

    # Sinkhorn 迭代
    for iteration in range(max_iter):
        # 行标准化
        row_sums = M.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=eps)
        M = M / row_sums
        
        # 列标准化
        col_sums = M.sum(dim=0, keepdim=True)
        col_sums = torch.clamp(col_sums, min=eps)
        M = M / col_sums
        
        # 收敛检测（行列和偏差）
        row_error = torch.max(torch.abs(row_sums.squeeze() - 1.0))
        col_error = torch.max(torch.abs(col_sums.squeeze() - 1.0))
        if row_error < tol and col_error < tol:
            break

    # 最终归一化（防护）
    M = _row_normalize_tensor(M, eps=eps)
    col_sums = M.sum(dim=0, keepdim=True)
    M = M / torch.clamp(col_sums, min=eps)
    return M

def row_normalize_tensor(mat: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-12) -> torch.Tensor:
    """Tensor版本的行归一化函数"""
    M = mat.clone()
    if mask is not None:
        M = torch.where(mask, M, 0.0)
    rowsum = M.sum(dim=1, keepdim=True)
    rowsum = torch.clamp(rowsum, min=eps)
    M = M / rowsum
    return M

def soft_topk_normalize_tensor(row: torch.Tensor, k: int, temperature: float = 1.0, 
                              hard: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """
    将一行概率分布限制为最多K个显著元素（soft）(Tensor版本)。
    row: shape (M,) tensor; 不含 depot 列（一般 = 客户子集）
    k: 最大车辆数
    temperature: softmax温度（越小越尖锐）
    hard: True 则硬 top-k (其余置eps), False 则使用温度放大 top-k 与缩小其余
    """
    M = row.shape[0]
    device = row.device
    
    if k >= M:
        # 无需截断
        r = row.clone()
        s = r.sum()
        return r / (s if s > eps else M)

    # 找 top-k 索引
    _, idx = torch.topk(row, k)
    topk_mask = torch.zeros(M, dtype=torch.bool, device=device)
    topk_mask[idx] = True

    r = torch.zeros_like(row)

    if hard:
        # 仅保留 top-k，其余置 eps
        r[topk_mask] = row[topk_mask]
        r[~topk_mask] = eps
    else:
        # soft: top-k 放大，非topk 缩小
        scale_large = 1.0
        scale_small = torch.exp(torch.tensor(-5.0 / temperature, device=device))
        r[topk_mask] = row[topk_mask] * scale_large
        r[~topk_mask] = torch.clamp(row[~topk_mask] * scale_small, min=eps)

    # 归一化
    s = r.sum()
    if s <= eps:
        r = torch.ones(M, device=device) / M
    else:
        r = r / s
    return r

# %% Tensor版本的主函数 
## =====================================================================================================
def gradient_assignment_vrp_KL_Sinkhorn_with_local_search_tensor(
    C: torch.Tensor, 
    G: torch.Tensor,
    depot: int = 0,
    K: int = 2,
    eta_cc: float = 1.0,       # 客户-客户梯度温度
    alpha_cc: float = 0.5,     # 客户-客户注入比例
    eta_depot: float = 1.0,    # depot->客户 梯度温度
    alpha_depot: float = 0.7,  # depot行注入比例
    sinkhorn_max_iter: int = 100,
    forbid_self_loop: bool = True,
    force_return_to_depot: bool = True,
    min_return_mass: float = 1e-3,
    depot_topk_hard: bool = False,
    depot_topk_temperature: float = 1.0,
    eps: float = 1e-12,
    # 新增局部搜索参数
    enable_local_search: bool = False,    # 是否启用局部搜索
    local_search_moves: int = 3,         # 局部搜索的最大移动次数
    local_transfer_ratio: float = 0.3,     # 局部搜索中权重转移的比例
    local_min_weight: float = 0.02,        # 局部搜索中参与转移的最小权重阈值
    local_improvement_threshold: float = 1e-5,  # 局部改进的阈值
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    基于 KL + Sinkhorn 的 VRP 梯度指派更新，带可选局部搜索增强 (Tensor版本)
    
    这是numpy版本gradient_assignment_vrp_KL_Sinkhorn_with_local_search的完全等价tensor实现。
    每个步骤都与numpy版本精确对应，确保数值结果一致。
    
    参数:
    - C: torch.Tensor - 当前连接矩阵 (N, N)
    - G: torch.Tensor - 梯度矩阵 (N, N)
    - 其他参数与numpy版本完全一致
    
    返回:
    - C_new: torch.Tensor - 更新后的连接矩阵
    - info: Dict - 包含更新信息的字典
    """
    # 🔧 修改：确保输入类型与numpy版本对应
    C = C.to(torch.float64)  # 对应 np.asarray(C, dtype=float)
    G = G.to(torch.float64)  # 对应 np.asarray(G, dtype=float)
    N = C.shape[0]
    device = C.device
    
    assert C.shape == (N, N), f"C shape should be (N, N), got {C.shape}"
    assert G.shape == (N, N), f"G shape should be (N, N), got {G.shape}"
    assert depot >= 0 and depot < N, f"depot should be in [0, {N-1}], got {depot}"

    clients = [i for i in range(N) if i != depot]

    # ---- Step 1: 客户-客户 KL + exp 梯度步 （完全对应numpy版本）
    # 🔧 修改：使用更精确的tensor索引，对应 np.ix_(clients, clients)
    clients_idx = torch.tensor(clients, device=device, dtype=torch.long)
    Cc = C[clients_idx][:, clients_idx]  # 对应 C[np.ix_(clients, clients)]
    Gc = G[clients_idx][:, clients_idx]  # 对应 G[np.ix_(clients, clients)]
    
    Sc = Cc * torch.exp(-eta_cc * Gc)  # 对应 Cc * np.exp(-eta_cc * Gc)
    Sc = torch.clamp(Sc, min=eps)      # 对应 np.maximum(Sc, eps)

    # mask 禁自环 - 完全对应numpy版本
    if forbid_self_loop:
        mask_c = torch.ones_like(Sc, dtype=torch.bool, device=device)
        mask_c.fill_diagonal_(False)  # 对应 np.fill_diagonal(mask_c, False)
        Sc = torch.where(mask_c, Sc, torch.tensor(0.0, device=device))  # 对应 np.where(mask_c, Sc, 0.0)
    else:
        mask_c = None

    # Sinkhorn -> 双随机近似
    Xc = sinkhorn_doubly_stochastic_tensor(
        Sc, mask=mask_c, max_iter=sinkhorn_max_iter, tol=1e-9, eps=eps
    )

    # ---- Step 2: 混合客户子块回全矩阵（完全对应numpy版本）
    X_full = C.clone()  # 对应 C.copy()
    # 🔧 修改：确保索引操作完全对应 X_full[np.ix_(clients, clients)] = Xc
    X_full[clients_idx[:, None], clients_idx] = Xc
    C_mid = (1 - alpha_cc) * C + alpha_cc * X_full

    # ---- Step 3: depot->客户 soft top-k（完全对应numpy版本）
    depot_row = C_mid[depot, clients_idx]   # 对应 C_mid[depot, clients]
    depot_grad = G[depot, clients_idx]      # 对应 G[depot, clients]
    
    # KL 指数步 - 完全对应numpy版本
    depot_step = depot_row * torch.exp(-eta_depot * depot_grad)  # 对应 depot_row * np.exp(-eta_depot * depot_grad)
    depot_step = torch.clamp(depot_step, min=eps)                # 对应 np.maximum(depot_step, eps)
    depot_step = depot_step / depot_step.sum()                   # 对应 depot_step / depot_step.sum()

    # soft top-k 限制车辆数
    depot_step = soft_topk_normalize_tensor(
        depot_step, k=K, temperature=depot_topk_temperature, hard=depot_topk_hard, eps=eps
    )

    # 注入回 depot 行
    depot_new_row = (1 - alpha_depot) * depot_row + alpha_depot * depot_step

    # 合成 depot 行（保持自身对角=0）
    C_mid[depot, clients_idx] = depot_new_row  # 对应 C_mid[depot, clients] = depot_new_row
    C_mid[depot, depot] = 0.0

    # ---- Step 4: 客户行强制"剩余质量 = 回 depot"（完全对应numpy版本）
    if force_return_to_depot:
        for i in clients:
            # 当前行
            row_client = C_mid[i, :]               # 对应 C_mid[i, :]
            # 去客户的总和
            to_clients = row_client[clients_idx].sum()  # 对应 row_client[clients].sum()
            # 目标行和=1: 让 C[i,depot] = 1 - to_clients（若负则截断）
            ret = 1.0 - to_clients
            if ret < min_return_mass:
                ret = min_return_mass  # 防止完全不回 depot
                # 再缩放客户子块 - 对应numpy版本的max(to_clients, eps)
                scale = (1.0 - ret) / torch.clamp(to_clients, min=eps)
                C_mid[i, clients_idx] = row_client[clients_idx] * scale  # 对应 C_mid[i, clients] = row_client[clients] * scale
            C_mid[i, depot] = ret

        # 行再归一 - 对应 row_normalize(C_mid, mask=None, eps=eps)
        C_mid = row_normalize_tensor(C_mid, mask=None, eps=eps)
    else:
        # 简单行归一 - 对应 row_normalize(C_mid, eps=eps)
        C_mid = row_normalize_tensor(C_mid, eps=eps)

    # ---- Step 5: 新增局部搜索阶段（完全对应numpy版本）----
    local_improvements = 0 
    if enable_local_search and len(clients) > 1:
        def local_refinement_tensor(matrix: torch.Tensor, grad_matrix: torch.Tensor):
            """局部搜索微调：完全对应numpy版本的local_refinement"""
            C_local = matrix.clone()  # 对应 matrix.copy()
            improvements = 0
            
            for move in range(local_search_moves):
                best_improvement = 0
                best_move = None
                
                # 寻找最有潜力的调整 - 完全对应numpy版本
                for i in clients:
                    row = C_local[i, clients_idx]      # 对应 C_local[i, clients]
                    grad_row = grad_matrix[i, clients_idx]  # 对应 grad_matrix[i, clients]
                    
                    # 🔧 修改：完全对应numpy版本的argsort操作
                    n_candidates = min(3, len(clients))
                    
                    # 对应 np.argsort(row)[-min(3, len(clients)):]
                    _, high_weight_indices = torch.topk(row, n_candidates)
                    # 对应 np.argsort(grad_row)[:min(3, len(clients))]
                    _, low_grad_indices = torch.topk(-grad_row, n_candidates)
                    
                    for hw_idx in high_weight_indices:
                        for lg_idx in low_grad_indices:
                            hw_idx_val = hw_idx.item()
                            lg_idx_val = lg_idx.item()
                            if hw_idx_val != lg_idx_val and row[hw_idx_val] > local_min_weight:
                                # 计算潜在改进（基于梯度差异）
                                potential = grad_row[hw_idx_val] - grad_row[lg_idx_val]
                                if potential > best_improvement:
                                    best_improvement = potential
                                    # 🔧 修改：这里的索引应该是全局索引，对应numpy版本的clients[hw_idx], clients[lg_idx]
                                    best_move = (i, clients[hw_idx_val], clients[lg_idx_val])
                
                # 如果找到改进，应用它 - 完全对应numpy版本
                if best_move and best_improvement > local_improvement_threshold:
                    i, j_from, j_to = best_move
                    # 对应 min(local_transfer_ratio * C_local[i, j_from], C_local[i, j_from] * 0.5)
                    transfer = torch.min(
                        torch.tensor(local_transfer_ratio, device=device, dtype=torch.float64) * C_local[i, j_from],
                        C_local[i, j_from] * 0.5
                    )
                    C_local[i, j_from] -= transfer
                    C_local[i, j_to] += transfer
                    improvements += 1
                else:
                    break  # 没有找到改进，停止
            
            return C_local, improvements
        
        # 应用局部搜索（只对客户部分）
        C_refined, local_improvements = local_refinement_tensor(C_mid, G)
        
        if local_improvements > 0:
            def apply_stability_constraint_tensor(C_old: torch.Tensor, C_new: torch.Tensor, 
                                                 max_change_ratio: float = 0.1) -> torch.Tensor:
                """应用稳定性约束防止过大变化 (完全对应numpy版本)"""
                change = torch.abs(C_new - C_old)
                max_allowed_change = max_change_ratio * torch.abs(C_old + eps)
                
                # 限制过大的变化
                excessive_change = change > max_allowed_change
                C_constrained = torch.where(
                    excessive_change,
                    C_old + torch.sign(C_new - C_old) * max_allowed_change,
                    C_new
                )
                return C_constrained
            
            # 保存原始C_mid用于稳定性约束
            C_original = C_mid.clone()  # 对应 C_original = C_mid.copy()
            
            # 保持depot相关的约束不变
            C_mid = C_refined.clone()   # 对应 C_mid = C_refined.copy()
            
            # 应用稳定性约束，防止过大变化
            C_mid = apply_stability_constraint_tensor(C_original, C_mid, max_change_ratio=0.1)
            
            # 重新应用depot约束 - 完全对应numpy版本
            if force_return_to_depot:
                for i in clients:
                    row_client = C_mid[i, :]
                    to_clients = row_client[clients_idx].sum()
                    ret = torch.clamp(torch.tensor(1.0 - to_clients, device=device), min=min_return_mass)
                    
                    if ret > min_return_mass and to_clients > eps:
                        scale = (1.0 - ret) / to_clients
                        C_mid[i, clients_idx] = row_client[clients_idx] * scale
                    C_mid[i, depot] = ret
            
            # 重新归一化 - 对应 row_normalize(C_mid, mask=None, eps=eps)
            C_mid = row_normalize_tensor(C_mid, mask=None, eps=eps)

    # ---- Step 6: 数值清理（完全对应numpy版本）----
    C_mid[depot, depot] = 0.0

    return C_mid, {
        "Xc_client_block": Xc,
        "depot_row_new": depot_new_row,
        "num_active_starts_soft": float((depot_new_row > 1e-6).sum().item()),
        "local_search_enabled": enable_local_search,
        "local_improvements": local_improvements,
        "method": "KL_Sinkhorn_with_local_search_tensor"
    }