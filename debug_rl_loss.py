#  %% [markdown]
## 导入必要的库
## =====================================================================================================
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

# 设置随机种子
torch.manual_seed(42)   
np.random.seed(42)


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"使用设备: {device}")


# %% [markdown]
## 导入真实函数
# 如果需要加载真实的函数，可以在这里从项目中导入 
## =====================================================================================================
import sys
import os

# 修改: 添加difusco路径到sys.path
difusco_path = '/home/yuepeng/codes/difusco_cross_pro/difusco'
if difusco_path not in sys.path:
    sys.path.append(difusco_path)
    print(f"已将 {difusco_path} 添加到 sys.path")

try:
    # 尝试从项目导入真实的函数
    from difusco.pl_tsp_model import (
        calculate_euclidean_distance_batch,
        calculate_tour_cost_batch,
        calculate_tour_cost_batch_pomo,
        greedy_solver_batch_pomo
    )
    print("成功导入项目中的真实函数")
    USE_REAL_FUNCTIONS = True
except ImportError as e:
    print(f"无法导入项目函数: {e}")
    print("将使用上面定义的简化版本")
    USE_REAL_FUNCTIONS = False

# %%
# 测试用的 TSPModel 类
## =====================================================================================================
class MockTrainer:
    def __init__(self):
        self.max_epochs = 100

class TestTSPModel:
    def __init__(self):
        # 强化学习相关参数
        self.rl_loss_weight = 0.1
        self.rl_baseline_decay = 0.95
        self.rl_baseline = None
        self.pomo_temperature = 0.1
        self.rl_failure_count = 0
        self.rl_skip_on_error = True
        
        # 数值稳定性参数
        self.max_logit_value = 50.0
        self.min_prob_value = 1e-8
        self.max_advantage = 20.0
        
        # 调试参数
        self.rl_debug = False
        self.add_prior = True
        
        # 模拟训练状态
        self.current_epoch = 10
        self.trainer = MockTrainer()
    
    def _compute_ground_truth_costs(self, gt_tour, distance_matrices):
        """计算真实最优路径的成本"""
        gt_tours_list = []
        for b in range(gt_tour.shape[0]):
            gt_tour_b = gt_tour[b].cpu().numpy().tolist()
            gt_tour_b.append(gt_tour_b[0])  # 添加回到起点
            gt_tours_list.append(gt_tour_b)
        
        gt_costs = calculate_tour_cost_batch(gt_tours_list, distance_matrices)
        return gt_costs
    
    def _compute_rewards_and_advantages(self, pred_costs, gt_costs, batch_size, current_problem_type):
        """计算奖励和优势函数"""
        # 扩展gt_costs以匹配POMO维度
        if current_problem_type == "TSP":
            num_starts = pred_costs.shape[0] // batch_size
        else:
            num_starts = pred_costs.shape[0] // batch_size
        
        gt_costs_expanded = gt_costs.unsqueeze(1).expand(-1, num_starts).reshape(-1)
        
        # 计算奖励 (负的相对成本差异)
        cost_diff = pred_costs - gt_costs_expanded
        relative_cost_diff = cost_diff / (gt_costs_expanded + self.min_prob_value)
        
        # 限制相对成本差异的范围以防止极端值
        relative_cost_diff = torch.clamp(relative_cost_diff, min=-10.0, max=10.0)
        rewards = -relative_cost_diff  # 成本越低，奖励越高
        
        # 对于POMO，选择每个样本中最好的路径来计算基线
        rewards_reshaped = rewards.reshape(batch_size, num_starts)
        best_rewards = torch.max(rewards_reshaped, dim=1)[0]
        
        # 更新基线 (使用指数移动平均)
        current_baseline = best_rewards.mean().detach()
        if torch.isnan(current_baseline) or torch.isinf(current_baseline):
            current_baseline = self.rl_baseline if self.rl_baseline is not None else torch.tensor(0.0)
        
        if self.rl_baseline is None:
            self.rl_baseline = current_baseline
        else:
            self.rl_baseline = self.rl_baseline_decay * self.rl_baseline + (1 - self.rl_baseline_decay) * current_baseline
        
        # 确保基线值有效
        if torch.isnan(self.rl_baseline) or torch.isinf(self.rl_baseline):
            self.rl_baseline = torch.tensor(0.0)
        
        # 计算优势函数 (奖励减去基线)
        advantages = rewards - self.rl_baseline
        
        # 限制优势函数的范围
        advantages = torch.clamp(advantages, min=-self.max_advantage, max=self.max_advantage)
        
        return rewards, advantages
    
    def _compute_rl_metrics(self, rl_loss, rewards, pred_costs, gt_costs, current_problem_type, batch_size):
        """计算并返回强化学习相关指标"""
        if current_problem_type == "TSP":
            num_starts = pred_costs.shape[0] // batch_size
        else:
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
            'pomo_temperature': self.pomo_temperature,
            'rl_failure_count': float(self.rl_failure_count)
        }
        
        return metrics
    
    def _handle_rl_computation_error(self, error, device):
        """处理强化学习计算错误"""
        self.rl_failure_count += 1
        error_msg = f"强化学习损失计算失败 ({self.rl_failure_count}): {error}"
        
        if self.rl_skip_on_error:
            print(f"WARNING: {error_msg}")
            rl_loss = torch.tensor(0.0, device=device)
            
            # 返回基本指标
            metrics = {
                'rl_loss': 0.0,
                'avg_reward': 0.0,
                'best_reward': 0.0,
                'baseline': 0.0,
                'avg_pred_cost': 0.0,
                'best_pred_cost': 0.0,
                'avg_gt_cost': 0.0,
                'cost_gap_percent': 0.0,
                'best_cost_gap_percent': 0.0,
                'pomo_temperature': self.pomo_temperature,
                'rl_failure_count': float(self.rl_failure_count)
            }
            
            return rl_loss, metrics
        else:
            # 不跳过错误，重新抛出异常
            raise RuntimeError(error_msg)
    
    def compute_reinforcement_learning_loss(self, x0_pred, points, gt_tour, current_problem_type, batch_idx):
        """计算强化学习损失函数"""
        device = points.device
        
        try:
            # 第一步：输入验证和预处理
            if torch.isnan(x0_pred).any() or torch.isinf(x0_pred).any():
                raise ValueError("x0_pred包含nan或inf")
            
            # 限制x0_pred的范围以防止softmax溢出
            x0_pred_clamped = torch.clamp(x0_pred, min=-self.max_logit_value, max=self.max_logit_value)
            
            # 计算softmax概率
            x0_pred_prob = x0_pred_clamped.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            
            if torch.isnan(x0_pred_prob).any() or torch.isinf(x0_pred_prob).any():
                raise ValueError("x0_pred_prob包含nan或inf")
            
            # 取边存在的概率 (第二个类别)
            adj_prob_matrix = x0_pred_prob[:, :, :, 1]  # shape: (batch_size, num_nodes, num_nodes)
            
            # 确保概率矩阵在有效范围内
            adj_prob_matrix = torch.clamp(adj_prob_matrix, min=self.min_prob_value, max=1.0 - self.min_prob_value)
            
            # 第二步：计算距离矩阵
            distance_matrices = calculate_euclidean_distance_batch(points[:, :, :2])
            
            if torch.isnan(distance_matrices).any() or torch.isinf(distance_matrices).any():
                raise ValueError("distance_matrices包含nan或inf")
            
            # 第三步：使用POMO求解器生成路径 
            pred_tours, log_probs = greedy_solver_batch_pomo(
                adj_prob_matrix, 
                temperature=self.pomo_temperature, 
                points_with_features=points, 
                problem_type=current_problem_type,
                add_prior=self.add_prior, 
                distance_matrices=distance_matrices, 
                test_mode=False,
                current_epoch=self.current_epoch, 
                max_epochs=self.trainer.max_epochs if self.trainer else None,
                debug=self.rl_debug
            )
            
            if log_probs is None or torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                raise ValueError("求解器返回的log_probs无效")
            
            # 第四步：计算预测路径成本
            pred_costs = calculate_tour_cost_batch_pomo(pred_tours, distance_matrices, problem_type=current_problem_type)
            
            if torch.isnan(pred_costs).any() or torch.isinf(pred_costs).any():
                raise ValueError("pred_costs包含nan或inf")
            
            # 第五步：计算真实最优路径成本
            gt_costs = self._compute_ground_truth_costs(gt_tour, distance_matrices)
            
            if torch.isnan(gt_costs).any() or torch.isinf(gt_costs).any() or (gt_costs <= 0).any():
                raise ValueError("gt_costs包含无效值")
            
            # 第六步：计算奖励和优势
            rewards, advantages = self._compute_rewards_and_advantages(
                pred_costs, gt_costs, points.shape[0], current_problem_type
            )
            
            if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                raise ValueError("rewards包含nan或inf")
            
            # 第七步：计算REINFORCE损失
            rl_loss_raw = -(log_probs * advantages.detach()).mean()
            
            if torch.isnan(rl_loss_raw) or torch.isinf(rl_loss_raw):
                raise ValueError("rl_loss_raw无效")
            
            rl_loss = rl_loss_raw
            
            # 重置失败计数（成功计算）
            self.rl_failure_count = 0
            
            # 第八步：计算指标
            metrics = self._compute_rl_metrics(
                rl_loss, rewards, pred_costs, gt_costs, current_problem_type, points.shape[0]
            )
            
            return rl_loss, metrics
            
        except Exception as e:
            # 错误处理
            return self._handle_rl_computation_error(e, device)

print("TestTSPModel 类定义完成")

# %% [markdown]
## 创建测试数据函数
## =====================================================================================================
def create_test_data(batch_size=2, num_nodes=5, problem_type="TSP"):
    """创建测试数据"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建随机点坐标
    if problem_type == "TSP":
        # TSP只需要2D坐标
        points = torch.rand(batch_size, num_nodes, 2, device=device)
    else:
        # VRP需要7维特征 [x, y, demand, early_tw, late_tw, route_open, length_limit]
        points = torch.zeros(batch_size, num_nodes, 7, device=device)
        points[:, :, :2] = torch.rand(batch_size, num_nodes, 2, device=device)  # 坐标
        points[:, :, 2] = torch.rand(batch_size, num_nodes, device=device) * 0.3  # 需求
        points[:, 0, 2] = 0  # depot需求为0
        points[:, :, 3] = torch.zeros(batch_size, num_nodes, device=device)  # 早期时间窗
        points[:, :, 4] = torch.ones(batch_size, num_nodes, device=device) * 10  # 晚期时间窗
        points[:, :, 5] = torch.zeros(batch_size, num_nodes, device=device)  # 开放路径标志
        points[:, :, 6] = torch.ones(batch_size, num_nodes, device=device) * 5  # 长度限制
    
    # 创建模拟的扩散模型输出 x0_pred
    # 形状: (batch_size, 2, num_nodes, num_nodes)
    # 第一个通道是"无边"的logit，第二个通道是"有边"的logit
    x0_pred = torch.randn(batch_size, 2, num_nodes, num_nodes, device=device)
    
    # 创建真实路径 (简单的顺序路径)
    gt_tour = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)
    for b in range(batch_size):
        gt_tour[b] = torch.arange(num_nodes, device=device)
    
    return points, x0_pred, gt_tour

# 创建测试数据
batch_size = 2
num_nodes = 5
problem_type = "TSP"
batch_idx = 0

points, x0_pred, gt_tour = create_test_data(batch_size, num_nodes, problem_type)

print(f"测试数据创建完成:")
print(f"- 批次大小: {batch_size}")
print(f"- 节点数量: {num_nodes}")
print(f"- 问题类型: {problem_type}")
print(f"- points形状: {points.shape}")
print(f"- x0_pred形状: {x0_pred.shape}")
print(f"- gt_tour形状: {gt_tour.shape}")
print(f"- 设备: {points.device}")


# %% [markdown]
## 从rl_debug.pkl导入调试数据
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
batch_idx = 0
x0_pred = x0_pred[batch_idx:batch_idx+1]  # 保持维度    
points = points[batch_idx:batch_idx+1]
gt_tour = gt_tour[batch_idx:batch_idx+1]
adj_matrix = adj_matrix[batch_idx:batch_idx+1]

print(f"\n分析第 {batch_idx} 个案例:")
print(f"- x0_pred形状: {x0_pred.shape}")
print(f"- points形状: {points.shape}")
print(f"- gt_tour形状: {gt_tour.shape}")

# %% [markdown]
## 将adj_matrix转换为x0_pred格式
adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()  # [1,51,51,2]
adj_matrix_onehot = adj_matrix_onehot.permute(0,3,1,2)  # [1,2,51,51]
print(adj_matrix_onehot.shape)  # 验证形状
# %% [markdown]
# 构建优化问题，找到合理的更新方向
import numpy as np
from scipy.optimize import linprog

def calculate_fully_constrained_update(C, g, budget, rank_penalty=100.0, norm_penalty=50.0, epsilon=1e-4):
    """
    计算最优更新方向Delta，同时考虑排序翻转和行和归一化。

    Args:
        C (np.ndarray): 当前连接矩阵 (N x N)。
        g (np.ndarray): 梯度 (N x N)。
        budget (float): L1范数总预算。
        rank_penalty (float): 违反排序翻转的惩罚权重。
        norm_penalty (float): 违反归一化的惩罚权重。
        epsilon (float): 排序翻转的最小间隔。

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): (最优更新矩阵Delta, 排序松弛变量, 归一化松弛变量)
    """
    N = C.shape[0]
    num_vars_delta = N * N

    # --- 1. 识别需要翻转的边 ---
    rank_flip_constraints = []
    for i in range(N):
        g_row = g[i, :].copy()
        g_row[i] = -np.inf
        u = np.argmax(g_row)
        g_row[i] = np.inf
        v = np.argmin(g_row)

        if g[i, u] > 0 and g[i, v] < 0 and C[i, u] < C[i, v]:
            rank_flip_constraints.append({'node': i, 'u': u, 'v': v})
    
    num_vars_t = num_vars_delta  # for L1 norm
    num_vars_s = len(rank_flip_constraints)  # for rank flipping slack
    num_vars_p = N  # for normalization positive deviation
    num_vars_n = N  # for normalization negative deviation

    # --- 2. 构建优化问题 ---
    # 变量x: [Δ_flat, t_flat, s_flat, p_flat, n_flat]
    total_vars = num_vars_delta + num_vars_t + num_vars_s + num_vars_p + num_vars_n
    
    # 目标函数: min(-g·Δ + W_rank·s + W_norm·(p+n))
    c_obj = np.concatenate([
        -g.flatten(),
        np.zeros(num_vars_t),
        np.full(num_vars_s, rank_penalty),
        np.full(num_vars_p, norm_penalty),
        np.full(num_vars_n, norm_penalty)
    ])

    # === 构建约束 ===
    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    # 约束: L1范数处理 |Δ_ij| <= t_ij
    for i in range(num_vars_delta):
        # Δ_ij - t_ij <= 0
        row_a = np.zeros(total_vars)
        row_a[i] = 1
        row_a[i + num_vars_delta] = -1
        A_ub.append(row_a)
        b_ub.append(0)
        # -Δ_ij - t_ij <= 0
        row_b = np.zeros(total_vars)
        row_b[i] = -1
        row_b[i + num_vars_delta] = -1
        A_ub.append(row_b)
        b_ub.append(0)

    # 约束: L1预算 sum(t_ij) <= budget
    row_budget = np.zeros(total_vars)
    row_budget[num_vars_delta : num_vars_delta + num_vars_t] = 1
    A_ub.append(row_budget)
    b_ub.append(budget)

    # 约束: 排序翻转 (软) Δ_iv - Δ_iu - s_k <= C_iu - C_iv - epsilon
    idx_s_start = num_vars_delta + num_vars_t
    for k, const in enumerate(rank_flip_constraints):
        i, u, v = const['node'], const['u'], const['v']
        row_rank = np.zeros(total_vars)
        row_rank[i * N + v] = 1
        row_rank[i * N + u] = -1
        row_rank[idx_s_start + k] = -1
        A_ub.append(row_rank)
        b_ub.append(C[i, u] - C[i, v] - epsilon)

    # 约束: 行和归一化 (软) sum_j(Δ_ij) - p_i + n_i = 1 - sum_j(C_ij)
    idx_p_start = idx_s_start + num_vars_s
    idx_n_start = idx_p_start + num_vars_p
    for i in range(N):
        row_norm = np.zeros(total_vars)
        row_norm[i * N : (i + 1) * N] = 1  # sum_j(Δ_ij)
        row_norm[idx_p_start + i] = -1     # -p_i
        row_norm[idx_n_start + i] = 1      # +n_i
        A_eq.append(row_norm)
        b_eq.append(1.0 - np.sum(C[i, :]))

    # 约束: 变量边界
    bounds = []
    c_flat = C.flatten()
    for i in range(num_vars_delta): bounds.append((-c_flat[i], 1 - c_flat[i])) # Δ
    for _ in range(num_vars_t): bounds.append((0, None)) # t
    for _ in range(num_vars_s): bounds.append((0, None)) # s
    for _ in range(num_vars_p): bounds.append((0, None)) # p
    for _ in range(num_vars_n): bounds.append((0, None)) # n

    # --- 3. 求解 ---
    # print("\n正在调用包含归一化约束的线性规划求解器...")
    result = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        # print("求解成功！")
        delta_flat = result.x[:num_vars_delta]
        slacks_rank = result.x[idx_s_start : idx_p_start]
        slacks_norm_p = result.x[idx_p_start : idx_n_start]
        slacks_norm_n = result.x[idx_n_start:]
        return delta_flat.reshape((N, N)), slacks_rank, (slacks_norm_p, slacks_norm_n)
    else:
        # print("求解失败！")
        print(result.message)
        return None, None, None

# --- 主程序 ---
if __name__ == "__main__":
    # 创建一个行和不为1的初始矩阵
    C = np.array([
        [0.0, 0.8, 0.1, 0.1], # sum = 1.0
        [0.7, 0.0, 0.2, 0.1], # sum = 1.0
        [0.3, 0.4, 0.0, 0.3], # sum = 1.0
        [0.1, 0.1, 0.8, 0.0]  # sum = 1.0
    ])
    # 为了演示效果，我们让其中一行的和不为1
    C[1, :] = C[1, :] * 0.8 # sum = 0.8

    g = np.array([
        [0.0, -1.2, 2.5, 0.1],
        [-0.5, 0.0, 1.8, 0.4],
        [1.5, -0.8, 0.0, 0.9],
        [0.2, 0.3, -2.1, 0.0]
    ])
    
    UPDATE_BUDGET = 1

    print("--- 初始状态 ---")
    print("当前连接矩阵 C:\n", C)
    print("初始行和:", np.sum(C, axis=1))
    
    delta, slacks_r, slacks_n = calculate_fully_constrained_update(
        C, g, budget=UPDATE_BUDGET, rank_penalty=100, norm_penalty=1)

    if delta is not None:
        C_new = C + delta
        
        print("\n--- 优化结果 ---")
        print("计算出的最优更新方向 Δ:\n", np.round(delta, 4))
        print(f"\n验证: Δ 的 L1 范数 = {np.sum(np.abs(delta)):.4f}")
        if slacks_r is not None and slacks_n is not None:
            print(f"排序松弛变量 s 的值: {np.round(slacks_r, 4)}")
            print(f"归一化松弛变量 p (正偏差): {np.round(slacks_n[0], 4)}")
            print(f"归一化松弛变量 n (负偏差): {np.round(slacks_n[1], 4)}")
        
        print("\n更新后的连接矩阵 C' = C + Δ:\n", np.round(C_new, 4))
        print("更新后的行和:", np.round(np.sum(C_new, axis=1), 4))
        print("  -> 可以看到行和被有效地拉回到了1.0附近。")
# %% [markdown]
## 运行强化学习损失计算
## =====================================================================================================
model = TestTSPModel()
model.rl_debug = False

# 让 x0_pred 可学习
x0_pred = adj_matrix_onehot
x0_pred_optim = torch.nn.Parameter(x0_pred.clone().detach())

# 设置优化器
optimizer = torch.optim.Adam([x0_pred_optim], lr=3e-2)
num_iters = 150

model.pomo_temperature = 0.1

from collections import defaultdict
import pandas as pd
from tqdm import tqdm

log = defaultdict(list)

print(f"开始 RL 微调（{num_iters} 步）")

# 添加进度条
pbar = tqdm(range(num_iters), desc="RL微调")
for step in pbar:
    optimizer.zero_grad()
    
    # 计算损失
    rl_loss, metrics = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred_optim,
        points=points,
        gt_tour=gt_tour,
        current_problem_type=problem_type,
        batch_idx=batch_idx
    )
    
    # 反向传播
    rl_loss.backward()
    optimizer.step()
    
    # 记录日志
    log["step"].append(step)
    log["loss"].append(rl_loss.item())
    for k, v in metrics.items():
        log[k].append(v)
    
    # 更新进度条描述
    pbar.set_postfix({
        'loss': f'{rl_loss.item():.4f}',
        'avg_cost': f'{metrics["avg_pred_cost"]:.4f}',
        'best_cost': f'{metrics["best_pred_cost"]:.4f}',
        'gap': f'{metrics["cost_gap_percent"]:.2f}%'
    })

print(f"\n微调完成！")

# 显示日志
df_log = pd.DataFrame(log)
print("\n前5步和后5步的训练日志:")
print(df_log.head())       
print(df_log.tail())       

# 保存更新后的参数
x0_pred_optim = x0_pred_optim.detach()
print(f"\n✅ 微调完成，x0_pred 已更新")

# %% [markdown]
## 绘制训练过程曲线
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

# %% [markdown]
## 评估基线 vs. 微调后性能
## =====================================================================================================
model.pomo_temperature = 0.

# 基线评估
with torch.no_grad():
    base_loss, base_metrics = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred.detach(),
        points=points,
        gt_tour=gt_tour,
        current_problem_type=problem_type,
        batch_idx=batch_idx
    )

# 微调后评估
with torch.no_grad():
    tuned_loss, tuned_metrics = model.compute_reinforcement_learning_loss(
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

# %% [markdown]
## 可视化邻接矩阵和路径
## =====================================================================================================
def plot_adj_matrix_and_route(adj_prob, points, title, ax_adj, ax_route):
    """绘制邻接矩阵和对应路径"""
    # 绘制邻接矩阵热力图
    im = ax_adj.imshow(adj_prob.cpu().numpy(), cmap='YlOrRd')
    ax_adj.set_title(f'Adjacency Matrix - {title}')
    plt.colorbar(im, ax=ax_adj)
    
    # 绘制路径图
    coords = points[:, :2].cpu().numpy()
    ax_route.scatter(coords[:, 0], coords[:, 1], s=100, c='blue', alpha=0.7)
    
    # 添加节点标签
    for j in range(len(coords)):
        ax_route.annotate(str(j), (coords[j, 0], coords[j, 1]), 
                       xytext=(5, 5), textcoords='offset points')
    
    # 根据邻接概率绘制边
    for i in range(len(coords)):
        for j in range(len(coords)):
            if adj_prob[i,j] > 0.3:  # 设置阈值
                alpha = float(adj_prob[i,j])  # 使用概率作为透明度
                ax_route.plot([coords[i,0], coords[j,0]], 
                          [coords[i,1], coords[j,1]], 
                          'g-', alpha=alpha, linewidth=2)
    
    ax_route.set_title(f'Route Graph - {title}')
    ax_route.set_aspect('equal')
    ax_route.grid(True, alpha=0.3)

def plot_ground_truth_route(gt_tour, points, title, ax_matrix, ax_route):
    """绘制ground truth最优路径"""
    # 确保gt_tour是numpy数组
    if torch.is_tensor(gt_tour):
        gt_tour = gt_tour.cpu().numpy()
    
    # 创建ground truth邻接矩阵
    num_nodes = len(gt_tour)
    gt_adj_matrix = torch.zeros(num_nodes, num_nodes)
    
    # 根据ground truth路径构建邻接矩阵
    for i in range(len(gt_tour)):
        current_node = int(gt_tour[i])
        next_node = int(gt_tour[(i + 1) % len(gt_tour)])
        gt_adj_matrix[current_node, next_node] = 1.0
    
    # 绘制邻接矩阵热力图
    im = ax_matrix.imshow(gt_adj_matrix.cpu().numpy(), cmap='YlOrRd', vmin=0, vmax=1)
    ax_matrix.set_title(f'Adjacency Matrix - {title}')
    plt.colorbar(im, ax=ax_matrix)
    
    # 绘制路径图
    coords = points[:, :2].cpu().numpy()
    ax_route.scatter(coords[:, 0], coords[:, 1], s=100, c='blue', alpha=0.7)
    
    # 添加节点标签
    for j in range(len(coords)):
        ax_route.annotate(str(j), (coords[j, 0], coords[j, 1]), 
                       xytext=(5, 5), textcoords='offset points')
    
    # 绘制ground truth路径
    for i in range(len(gt_tour)):
        start = coords[int(gt_tour[i])]
        end = coords[int(gt_tour[(i + 1) % len(gt_tour)])]
        ax_route.plot([start[0], end[0]], [start[1], end[1]], 'r-', alpha=0.8, linewidth=3)
    
    ax_route.set_title(f'Route Graph - {title}')
    ax_route.set_aspect('equal')
    ax_route.grid(True, alpha=0.3)

# 创建2x3子图布局
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 获取第一个样本的数据
sample_idx = 0
base_adj_prob = torch.softmax(x0_pred[sample_idx].permute(1,2,0), dim=-1)[:,:,1]
tuned_adj_prob = torch.softmax(x0_pred_optim[sample_idx].permute(1,2,0), dim=-1)[:,:,1]
sample_points = points[sample_idx]

# 绘制基线模型结果
plot_adj_matrix_and_route(base_adj_prob, sample_points, 
                         "Before Update", axes[0,0], axes[0,1])

# 绘制微调后结果
plot_adj_matrix_and_route(tuned_adj_prob, sample_points,
                         "After Update", axes[1,0], axes[1,1])

# Plot ground truth route
gt_path = gt_tour[sample_idx]
coords = sample_points[:, :2].detach().cpu().numpy()

for i in [axes[0,2], axes[1,2]]:
    i.scatter(coords[:, 0], coords[:, 1], s=100, c='blue', alpha=0.7)
    
    # Add node labels
    for j in range(len(coords)):
        i.annotate(str(j), (coords[j, 0], coords[j, 1]), 
                  xytext=(5, 5), textcoords='offset points')
    
    # Draw ground truth path
    for j in range(len(gt_path)):
        start = coords[gt_path[j]]
        end = coords[gt_path[(j + 1) % len(gt_path)]]
        i.plot([start[0], end[0]], [start[1], end[1]], 'r-', alpha=0.8, linewidth=2)
    
    i.set_title('Ground Truth Optimal Route')
    i.set_aspect('equal')
    i.grid(True, alpha=0.3) 

plt.tight_layout()
plt.show()
# %%
## 约束满足的测试函数
## =====================================================================================================

print("\n" + "="*80)
print("约束满足的测试函数")
print("="*80)

# 导入必要的函数 
from difusco.pl_tsp_model import simulate_vrp_execution, visualize_vrp_solution

def test_constraint_satisfaction(
    points,                  # 点集tensor [batch_size, num_nodes, features]
    gt_tour,                # Ground Truth路径tensor [batch_size, num_nodes]
    optimized_route,        # 2-opt优化后的路径列表 tensor(batch_size, num_nodes)
    gt_cost,               # Ground Truth路径成本
    optimized_cost,        # 2-opt优化后的路径成本
    problem_type,          # 问题类型
    USE_REAL_FUNCTIONS=True # 是否使用真实约束检查函数
):
    """测试2-opt得到的解是否满足约束
    
    Args:
        points: 点集tensor [batch_size, num_nodes, features]
        gt_tour: Ground Truth路径tensor [batch_size, num_nodes] 
        optimized_route: 2-opt优化后的路径列表
        gt_cost: Ground Truth路径成本
        optimized_cost: 2-opt优化后的路径成本
        problem_type: 问题类型
        USE_REAL_FUNCTIONS: 是否使用真实约束检查函数
        
    Returns:
        dict: 包含测试结果的字典
    """
    
    # 1. 准备测试数据
    # 将tensor转换为numpy数组
    points_np = points[0].cpu().numpy()  # shape: (num_nodes, 2或7)
    
    # 检查并准备特征数据
    if points_np.shape[1] == 2:
        # 只有2D坐标，需要构建7维特征用于VRP验证
        num_nodes = points_np.shape[0]
        points_with_features = np.zeros((num_nodes, 7))
        points_with_features[:, :2] = points_np  # 设置坐标
        points_with_features[:, 2] = np.concatenate([np.array([0]), np.random.rand(num_nodes-1) * 0.1])  # 需求
        points_with_features[:, 3] = np.zeros(num_nodes)  # 早期时间窗
        points_with_features[:, 4] = np.ones(num_nodes) * 10  # 晚期时间窗
        points_with_features[:, 5] = np.zeros(num_nodes)  # 开放路径标志
        points_with_features[:, 6] = np.ones(num_nodes) * 3.0  # 路径长度限制
        print(f"✓ 从2D坐标构建了完整的7维特征向量")
    else:
        num_nodes = points_np.shape[0]
        points_with_features = points_np
        print(f"✓ 使用现有的{points_np.shape[1]}维特征向量")
    
    # 2. 测试Ground Truth路径
    gt_route = gt_tour[0].cpu().numpy().tolist()
    print(f"\n【Ground Truth路径测试】")
    print(f"路径: {gt_route}")
    
    gt_violations = None
    try:
        if USE_REAL_FUNCTIONS:
            gt_execution_history = simulate_vrp_execution(
                tour=gt_route,
                points_with_features=points_with_features,
                problem_type=problem_type
            )
            
            gt_violations = gt_execution_history['constraint_violations']
            print(f"约束违反统计:")
            print(f"  - 容量约束违反: {gt_violations['capacity_violations']}")
            print(f"  - 时间窗约束违反: {gt_violations['time_window_violations']}")
            print(f"  - 路径长度约束违反: {gt_violations['length_violations']}")
            print(f"  - 总违反次数: {gt_violations['total_violations']}")
            
            if gt_violations['total_violations'] == 0:
                print("✅ Ground Truth路径满足所有约束")
            else:
                print("❌ Ground Truth路径存在约束违反")
                
        else:
            print("⚠️ 无法导入真实函数，跳过Ground Truth路径测试")
            
    except Exception as e:
        print(f"❌ Ground Truth路径测试失败: {e}")
    
    # 3. 测试2-opt优化路径
    print(f"\n【2-opt优化路径测试】")
    print(f"路径: {optimized_route}")
    # 检查optimized_route是否为tensor,如果是则转换为list
    if isinstance(optimized_route, torch.Tensor):
        optimized_route = optimized_route[0].cpu().numpy().tolist()
    
    opt_violations = None
    try:
        if USE_REAL_FUNCTIONS:
            opt_execution_history = simulate_vrp_execution(
                tour=optimized_route,
                points_with_features=points_with_features,
                problem_type=problem_type
            )
            
            opt_violations = opt_execution_history['constraint_violations']
            print(f"约束违反统计:")
            print(f"  - 容量约束违反: {opt_violations['capacity_violations']}")
            print(f"  - 时间窗约束违反: {opt_violations['time_window_violations']}")
            print(f"  - 路径长度约束违反: {opt_violations['length_violations']}")
            print(f"  - 总违反次数: {opt_violations['total_violations']}")
            
            if opt_violations['total_violations'] == 0:
                print("✅ 2-opt路径满足所有约束")
            else:
                print("❌ 2-opt路径存在约束违反")
                
                # 详细分析违反情况
                print(f"\n详细违反分析:")
                violations_details = opt_execution_history['violations']
                for i, step_violations in enumerate(violations_details):
                    if step_violations:
                        print(f"  步骤 {i}: {step_violations}")
                        
        else:
            print("⚠️ 无法导入真实函数，跳过2-opt路径测试")
            
    except Exception as e:
        print(f"❌ 2-opt路径测试失败: {e}")
    
    # 4. 可视化对比
    print(f"\n【可视化对比】")
    try:
        if USE_REAL_FUNCTIONS:
            # 可视化Ground Truth路径
            print("生成Ground Truth路径可视化...")
            fig_gt, _ = visualize_vrp_solution(
                points=points_np[:, :2],
                tour=gt_route,
                points_with_features=points_with_features,
                problem_type=problem_type,
                gt_cost=gt_cost,
                pred_cost=gt_cost,
                title_suffix="Ground Truth",
                show_constraints=True,
                figsize=(15, 10)
            )
            plt.show()
            
            # 可视化2-opt路径
            print("生成2-opt路径可视化...")
            fig_opt, _ = visualize_vrp_solution(
                points=points_np[:, :2],
                tour=optimized_route,
                points_with_features=points_with_features,
                problem_type=problem_type,
                gt_cost=gt_cost,
                pred_cost=optimized_cost,
                title_suffix="2-opt Optimized",
                show_constraints=True,
                figsize=(15, 10)
            )
            plt.show()
            
        else:
            print("⚠️ 无法导入真实函数，跳过可视化")
            
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
    
    # 5. 总结报告
    print(f"\n" + "="*50)
    print("测试总结报告")
    print("="*50)
    
    print(f"问题类型: {problem_type}")
    print(f"节点数量: {num_nodes}")
    print(f"Ground Truth成本: {gt_cost:.4f}")
    print(f"2-opt优化成本: {optimized_cost:.4f}")
    print(f"成本改进: {((gt_cost - optimized_cost) / gt_cost * 100):.2f}%")
    
    # 6. 返回测试结果
    results = {
        'problem_type': problem_type,
        'num_nodes': num_nodes,
        'gt_cost': gt_cost,
        'optimized_cost': optimized_cost,
        'improvement': ((gt_cost - optimized_cost) / gt_cost * 100),
        'gt_violations': gt_violations,
        'opt_violations': opt_violations,
        'points_with_features': points_with_features
    }
    
    if USE_REAL_FUNCTIONS:
        try:
            # 检查gt_violations和opt_violations是否为None
            if gt_violations is None or opt_violations is None:
                print("⚠️ 约束违反数据不可用")
                results['constraint_status'] = 'data_unavailable'
            else:
                # 安全地访问字典
                gt_total = gt_violations.get('total_violations', 0)
                opt_total = opt_violations.get('total_violations', 0)
                
                print(f"Ground Truth约束违反: {gt_total}")
                print(f"2-opt约束违反: {opt_total}")
                
                if gt_total == 0 and opt_total == 0:
                    print("✅ 所有解都满足约束条件")
                    results['constraint_status'] = 'all_satisfied'
                elif gt_total > 0:
                    print("❌ Ground Truth解本身就存在约束违反")
                    results['constraint_status'] = 'gt_violated'
                elif opt_total > 0:
                    print("❌ 2-opt解引入了约束违反")
                    results['constraint_status'] = 'opt_violated'
                else:
                    print("⚠️ 约束状态未知")
                    results['constraint_status'] = 'unknown'
                
        except Exception as e:
            print(f"⚠️ 无法获取约束违反统计: {str(e)}")
            results['constraint_status'] = 'error'
    else:
        print("⚠️ 无法进行约束验证（缺少真实函数）")
        results['constraint_status'] = 'no_check'
        
    return results
# 运行测试
# if __name__ == "__main__":
#     test_constraint_satisfaction(
#         points=points,
#         gt_tour=gt_tour,
#         optimized_route=optimized_route,
#         gt_cost=gt_cost,
#         optimized_cost=optimized_cost,
#         problem_type=problem_type
#     )

# %%
## OR-TOOLS求解器
## =====================================================================================================

print("\n" + "="*80)
print("OR-TOOLS求解器")
print("="*80)

# 尝试导入OR-Tools
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    print("✅ 成功导入OR-Tools")
    OR_TOOLS_AVAILABLE = True
except ImportError:
    print("❌ OR-Tools未安装，请运行: pip install ortools")
    OR_TOOLS_AVAILABLE = False

def extract_cvrp_data(points_with_features):
    """
    从points_with_features中提取CVRP问题数据
    
    根据VRPGraphDataset的定义，7维特征的含义是：
    - depot节点：[x, y, 0, 0, 0, 0, 0] （坐标 + 5个零值）
    - 客户节点：[x, y, demand, earlyTW, lateTW, route_open, length]
    
    Args:
        points_with_features: np.array of shape (num_nodes, 2-7) 
                            [x, y, demand, early_tw, late_tw, route_open, length_limit] 
    
    Returns:
        dict - 包含CVRP问题数据的字典
    """
    num_nodes = points_with_features.shape[0]
    
    # 提取各维度特征
    coordinates = points_with_features[:, :2]  # x, y坐标
    demands = points_with_features[:, 2]       # 需求量
    early_tw = points_with_features[:, 3]      # 早期时间窗
    late_tw = points_with_features[:, 4]       # 晚期时间窗
    route_open = points_with_features[:, 5]    # 路径开放标志
    length_limit = points_with_features[:, 6]  # 路径长度限制
    
    print(f"提取的特征信息:")
    print(f"  - 节点数量: {num_nodes}")
    print(f"  - 坐标范围: x[{coordinates[:, 0].min():.3f}, {coordinates[:, 0].max():.3f}], y[{coordinates[:, 1].min():.3f}, {coordinates[:, 1].max():.3f}]")
    print(f"  - 需求量范围: [{demands.min():.3f}, {demands.max():.3f}]")
    print(f"  - 早期时间窗范围: [{early_tw.min():.3f}, {early_tw.max():.3f}]")
    print(f"  - 晚期时间窗范围: [{late_tw.min():.3f}, {late_tw.max():.3f}]")
    print(f"  - 路径开放标志范围: [{route_open.min():.3f}, {route_open.max():.3f}]")
    print(f"  - 长度限制范围: [{length_limit.min():.3f}, {length_limit.max():.3f}]")
    
    # 计算距离矩阵（欧几里得距离，放大1000倍以避免浮点数问题）
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
                # 确保距离值有效且为正整数
                dist_scaled = max(1, int(dist * 1000))  # 最小距离为1，避免0距离
                distance_matrix[i][j] = dist_scaled
    
    # 将需求转换为整数（放大1000倍）
    demands_int = (demands * 1000).astype(int)
    
    # 检查需求数据有效性
    if np.any(np.isnan(demands_int)) or np.any(demands_int < 0):
        print("⚠️ 需求数据包含无效值，进行修正")
        demands_int = np.nan_to_num(demands_int, nan=0.0, posinf=0.0, neginf=0.0).astype(int)
    
    # 根据项目的设定，车辆初始容量为1.0（对应1000），需求通过减法更新载重
    # 这意味着车辆容量应该设置为满足所有需求的总和
    total_demand = np.sum(demands_int[1:])  # 排除depot（索引0）
    
    # 设置车辆容量
    # 根据模拟执行代码，车辆开始时载重为1.0，然后减去需求
    # 所以车辆容量应该是1000（对应1.0）
    vehicle_capacity = 1000
    
    # 计算最少需要的车辆数
    if total_demand > 0:
        min_vehicles = max(1, int(np.ceil(total_demand / vehicle_capacity)))
    else:
        min_vehicles = 1
    
    # 设置车辆数量上限（给一些冗余）
    max_vehicles = min(num_nodes - 1, min_vehicles + 3)
    
    print(f"车辆配置:")
    print(f"  - 总需求: {total_demand} (原始: {total_demand/1000:.3f})")
    print(f"  - 单车容量: {vehicle_capacity} (原始: {vehicle_capacity/1000:.3f})")
    print(f"  - 最少车辆数: {min_vehicles}")
    print(f"  - 最大车辆数: {max_vehicles}")
    
    # 检查时间窗和长度限制是否有效
    has_time_windows = (late_tw > early_tw).any() and (late_tw > 0).any()
    has_length_limits = (length_limit > 0).any()
    
    print(f"约束检查:")
    print(f"  - 时间窗约束: {'有效' if has_time_windows else '无效'}")
    print(f"  - 长度限制约束: {'有效' if has_length_limits else '无效'}")
    print(f"  - 路径开放标志: {'有效' if (route_open > 0).any() else '无效'}")
    
    cvrp_data = {
        'distance_matrix': distance_matrix,
        'demands': demands_int,
        'vehicle_capacities': [vehicle_capacity] * max_vehicles,
        'num_vehicles': max_vehicles,
        'depot': 0,
        'coordinates': coordinates,
        'original_demands': demands,
        'early_tw': early_tw,
        'late_tw': late_tw,
        'route_open': route_open,
        'length_limit': length_limit,
        'has_time_windows': has_time_windows,
        'has_length_limits': has_length_limits,
        'total_demand': total_demand,
        'min_vehicles': min_vehicles
    }
    
    return cvrp_data

def solve_cvrp_with_ortools(cvrp_data, time_limit_seconds=60):
    """
    使用OR-Tools求解CVRP问题，支持容量、时间窗和长度限制约束
    
    Args:
        cvrp_data: dict - CVRP问题数据
        time_limit_seconds: int - 求解时间限制（秒）
    
    Returns:
        dict - 求解结果
    """
    if not OR_TOOLS_AVAILABLE:
        return None
    
    # 创建路由索引管理器
    manager = pywrapcp.RoutingIndexManager(
        len(cvrp_data['distance_matrix']),
        cvrp_data['num_vehicles'],
        cvrp_data['depot']
    )
    
    # 创建路由模型
    routing = pywrapcp.RoutingModel(manager)
    
    # 定义距离回调函数
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(cvrp_data['distance_matrix'][from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # 设置路径成本
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 1. 容量约束
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(cvrp_data['demands'][from_node])
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        cvrp_data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )
    
    # 2. 时间窗约束（如果有效）
    if cvrp_data['has_time_windows']:
        print("  - 添加时间窗约束")
        
        # 时间窗约束的处理需要将距离矩阵转换为时间（这里假设距离=时间）
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(cvrp_data['distance_matrix'][from_node][to_node])
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # 计算时间窗约束的放大系数（与距离相同）
        early_tw_scaled = (cvrp_data['early_tw'] * 1000).astype(int)
        late_tw_scaled = (cvrp_data['late_tw'] * 1000).astype(int)
        
        # 检查数据有效性
        if np.any(np.isnan(early_tw_scaled)) or np.any(np.isnan(late_tw_scaled)):
            print("⚠️ 时间窗数据包含NaN，跳过时间窗约束")
            return None
        
        horizon = int(late_tw_scaled.max()) + 100000  # 时间范围上限
        
        routing.AddDimension(
            time_callback_index,
            horizon,  # allow waiting time
            horizon,  # maximum time per vehicle
            False,  # Don't force start cumul to zero
            'Time'
        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # 为每个节点设置时间窗
        for node in range(len(cvrp_data['early_tw'])):
            if node == 0:  # depot
                continue
            index = manager.NodeToIndex(node)
            
            # 确保传递给OR-Tools的是Python int类型
            early_time = int(early_tw_scaled[node])
            late_time = int(late_tw_scaled[node])
            
            # 检查时间窗的有效性
            if early_time >= late_time:
                print(f"⚠️ 节点{node}的时间窗无效: [{early_time}, {late_time}]，跳过")
                continue
                
            time_dimension.CumulVar(index).SetRange(early_time, late_time)
    
    # 3. 长度限制约束（如果有效）
    if cvrp_data['has_length_limits']:
        print("  - 添加长度限制约束")
        
        # 使用第一个客户节点的长度限制作为所有车辆的长度限制
        if len(cvrp_data['length_limit']) > 1:
            length_limit_scaled = int(cvrp_data['length_limit'][1] * 1000)
        else:
            length_limit_scaled = 3000
            
        # 确保长度限制是正整数
        length_limit_scaled = max(1000, length_limit_scaled)  # 最小长度限制
        
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            length_limit_scaled,  # vehicle maximum travel distance
            True,  # start cumul to zero
            'Distance'
        )
        
        distance_dimension = routing.GetDimensionOrDie('Distance')
        distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    # 设置求解参数
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(time_limit_seconds)
    
    # 求解
    print(f"正在求解CVRP问题...")
    print(f"  - 节点数: {len(cvrp_data['distance_matrix'])}")
    print(f"  - 车辆数: {cvrp_data['num_vehicles']}")
    print(f"  - 车辆容量: {cvrp_data['vehicle_capacities'][0]}")
    print(f"  - 时间窗约束: {'启用' if cvrp_data['has_time_windows'] else '禁用'}")
    print(f"  - 长度限制约束: {'启用' if cvrp_data['has_length_limits'] else '禁用'}")
    print(f"  - 时间限制: {time_limit_seconds}秒")
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        return parse_ortools_solution(manager, routing, solution, cvrp_data)
    else:
        print("❌ 未找到解决方案")
        return None

def parse_ortools_solution(manager, routing, solution, cvrp_data):
    """
    解析OR-Tools的求解结果
    """
    print(f"✅ 找到解决方案!")
    print(f"目标值: {solution.ObjectiveValue()}")
    
    # 提取路径
    routes = []
    total_distance = 0
    total_load = 0
    
    for vehicle_id in range(cvrp_data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        route_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            route_load += cvrp_data['demands'][node_index]
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        
        # 添加回到depot的节点
        route.append(manager.IndexToNode(index))
        
        # 只保存非空路径
        if len(route) > 2:  # 至少有depot -> node -> depot
            routes.append(route)
            total_distance += route_distance
            total_load += route_load
            
            print(f"车辆 {vehicle_id}: {' -> '.join(map(str, route))}")
            print(f"  路径距离: {route_distance}")
            print(f"  路径负载: {route_load}/{cvrp_data['vehicle_capacities'][0]}")
    
    result = {
        'routes': routes,
        'total_distance': total_distance,
        'total_load': total_load,
        'objective_value': solution.ObjectiveValue(),
        'num_vehicles_used': len(routes),
        'success': True
    }
    
    print(f"\n求解结果摘要:")
    print(f"  - 使用车辆数: {len(routes)}")
    print(f"  - 总距离: {total_distance}")
    print(f"  - 总负载: {total_load}")
    
    return result

def convert_gt_tour_to_routes(gt_tour, num_nodes):
    """
    将ground truth tour转换为路径格式
    """
    routes = []
    current_route = []
    
    for node in gt_tour:
        if node >= num_nodes:  # 超出节点范围，跳过
            continue
            
        if node == 0:  # depot
            if current_route:  # 如果当前路径不为空
                current_route.append(0)  # 添加回到depot
                routes.append(current_route)
                current_route = []
            current_route.append(0)  # 开始新路径
        else:
            current_route.append(node)
    
    # 处理最后一个路径
    if current_route and current_route != [0]:
        current_route.append(0)
        routes.append(current_route)
    
    return routes

def calculate_route_cost(route, distance_matrix):
    """计算路径成本"""
    cost = 0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i + 1]]
    return cost

def compare_solutions(solver_result, gt_routes, cvrp_data):
    """
    比较求解器结果与ground truth
    """
    print(f"\n" + "="*50)
    print("解决方案比较")
    print("="*50)
    
    if solver_result is None:
        print("❌ 无法进行比较（求解器未找到解）")
        return
    
    # 计算ground truth的成本
    gt_total_cost = 0
    gt_num_vehicles = len(gt_routes)
    
    print(f"Ground Truth 路径:")
    for i, route in enumerate(gt_routes):
        route_cost = calculate_route_cost(route, cvrp_data['distance_matrix'])
        gt_total_cost += route_cost
        print(f"  车辆 {i}: {' -> '.join(map(str, route))}")
        print(f"    成本: {route_cost}")
    
    print(f"\nOR-Tools 求解器路径:")
    solver_total_cost = solver_result['total_distance']
    solver_num_vehicles = solver_result['num_vehicles_used']
    
    for i, route in enumerate(solver_result['routes']):
        route_cost = calculate_route_cost(route, cvrp_data['distance_matrix'])
        print(f"  车辆 {i}: {' -> '.join(map(str, route))}")
        print(f"    成本: {route_cost}")
    
    # 比较结果
    print(f"\n比较结果:")
    print(f"  Ground Truth:")
    print(f"    - 车辆数: {gt_num_vehicles}")
    print(f"    - 总成本: {gt_total_cost}")
    print(f"  OR-Tools 求解器:")
    print(f"    - 车辆数: {solver_num_vehicles}")
    print(f"    - 总成本: {solver_total_cost}")
    
    cost_diff = gt_total_cost - solver_total_cost
    cost_improvement = (cost_diff / gt_total_cost) * 100 if gt_total_cost > 0 else 0
    
    print(f"  差异分析:")
    print(f"    - 成本差异: {cost_diff}")
    print(f"    - 成本改进: {cost_improvement:.2f}%")
    print(f"    - 车辆数差异: {gt_num_vehicles - solver_num_vehicles}")
    
    if cost_diff > 0:
        print(f"✅ OR-Tools找到了更好的解决方案")
    elif cost_diff < 0:
        print(f"❌ Ground Truth解决方案更好")
    else:
        print(f"➡️ 两个解决方案成本相同")
    
    return {
        'gt_cost': gt_total_cost,
        'solver_cost': solver_total_cost,
        'cost_improvement': cost_improvement,
        'gt_vehicles': gt_num_vehicles,
        'solver_vehicles': solver_num_vehicles
    }

def visualize_solution_comparison(cvrp_data, solver_result, gt_routes):
    """
    可视化解决方案比较
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    coordinates = cvrp_data['coordinates']
    
    # 绘制Ground Truth解
    max_routes = max(len(gt_routes), len(solver_result['routes']) if solver_result and solver_result['routes'] else 1)
    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, max_routes))
    
    ax1.scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=100, alpha=0.7, zorder=5)
    ax1.scatter(coordinates[0, 0], coordinates[0, 1], c='blue', s=200, marker='s', alpha=0.9, zorder=6, label='Depot')
    
    for i, route in enumerate(gt_routes):
        color = colors[i % len(colors)]
        for j in range(len(route) - 1):
            start = coordinates[route[j]]
            end = coordinates[route[j + 1]]
            ax1.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2, alpha=0.8)
    
    ax1.set_title('Ground Truth Solution')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制OR-Tools解
    if solver_result and solver_result.get('routes'):
        ax2.scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=100, alpha=0.7, zorder=5)
        ax2.scatter(coordinates[0, 0], coordinates[0, 1], c='blue', s=200, marker='s', alpha=0.9, zorder=6, label='Depot')
        
        for i, route in enumerate(solver_result['routes']):
            color = colors[i % len(colors)]
            for j in range(len(route) - 1):
                start = coordinates[route[j]]
                end = coordinates[route[j + 1]]
                ax2.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2, alpha=0.8)
        
        ax2.set_title('OR-Tools Solution')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'OR-Tools解不可用', ha='center', va='center', transform=ax2.transAxes, fontsize=16)
        ax2.set_title('OR-Tools Solution (Not Available)')
    
    plt.tight_layout()
    plt.show()
def main_cvrp_verification():
    """主验证函数"""
    print("开始CVRP最优性验证...")
    
    # 1. 提取CVRP数据
    points_np = points[0].cpu().numpy()
    if points_np.shape[1] < 7:
        print("❌ 数据不完整，无法进行CVRP验证")
        return
    
    cvrp_data = extract_cvrp_data(points_np)
    print(f"✅ 成功提取CVRP数据")
    
    # 2. 转换Ground Truth路径
    gt_tour_np = gt_tour[0].cpu().numpy()
    gt_routes = convert_gt_tour_to_routes(gt_tour_np, points_np.shape[0])
    print(f"✅ Ground Truth路径转换完成，共{len(gt_routes)}条路径")
    
    # 3. 使用OR-Tools求解
    if OR_TOOLS_AVAILABLE:
        solver_result = solve_cvrp_with_ortools(cvrp_data, time_limit_seconds=120)
    else:
        solver_result = None
        print("❌ OR-Tools不可用，跳过求解器验证")
    
    # 4. 比较结果
    comparison_result = compare_solutions(solver_result, gt_routes, cvrp_data)
    
    # 5. 可视化比较
    print(f"\n生成可视化比较图...")
    visualize_solution_comparison(cvrp_data, solver_result, gt_routes)
    
    return comparison_result

# 运行CVRP验证
if __name__ == "__main__":
    if problem_type == "CVRP" or "VRP" in problem_type:
        main_cvrp_verification()
    else:
        print(f"当前问题类型为 {problem_type}，CVRP验证模块仅支持VRP类问题")

# %%
## 约束验证模块：验证OR-Tools解和GT解的约束满足情况
## =====================================================================================================

print("\n" + "="*80)
print("约束验证模块")
print("="*80)
 
"""验证OR-Tools解和Ground Truth解的约束满足情况"""
print("开始验证OR-Tools解和Ground Truth解的约束...")

# 1. 提取数据
points_np = points[0].cpu().numpy()
gt_tour_np = gt_tour[0].cpu().numpy()

# 2. 获取OR-Tools解 
cvrp_data = extract_cvrp_data(points_np)
solver_result = solve_cvrp_with_ortools(cvrp_data, time_limit_seconds=120)
if solver_result and 'routes' in solver_result:
    ortools_routes = solver_result['routes']
    ortools_cost = solver_result.get('total_distance', 0) / 1000  # 转换回原始尺度
    print(f"✅ 成功获取OR-Tools解，成本为: {ortools_cost:.2f}")
else:
    print("❌ 无法获取有效的OR-Tools解")  

# %%
# 3. 调用约束验证函数
# 将numpy数组转换为tensor
points_tensor = torch.from_numpy(points_np).unsqueeze(0)  # [1, num_nodes, features]
gt_tour_tensor = torch.from_numpy(gt_tour_np).unsqueeze(0)  # [1, num_nodes]

# 将ortools_routes转换为tensor  
# 将多条路径拼接成一个长的list
ortools_routes_combined = []
for route in ortools_routes:
    # 如果不是第一条路径,去掉起点(因为已经在前一条路径的终点)
    if len(ortools_routes_combined) > 0:
        ortools_routes_combined.extend(route[1:])
    else:
        ortools_routes_combined.extend(route)
print(f"✓ 将OR-Tools的{len(ortools_routes)}条路径合并为单一路径: {ortools_routes_combined}")

ortools_routes_np = np.array(ortools_routes_combined)
# 转换为tensor并增加batch维度
ortools_routes_tensor = torch.from_numpy(ortools_routes_np).unsqueeze(0) 

results = test_constraint_satisfaction(
    points=points_tensor,
    gt_tour=gt_tour_tensor,
    optimized_route=ortools_routes_tensor,  # 使用OR-Tools的解
    gt_cost=gt_cost,
    optimized_cost=ortools_cost,
    problem_type=problem_type
)

print("\n约束验证结果:")
print(f"Ground Truth成本: {results['gt_cost']:.2f}")
print(f"OR-Tools解成本: {results['optimized_cost']:.2f}")
print(f"改进百分比: {results['improvement']:.2f}%")
print(f"约束状态: {results['constraint_status']}")

# %% [markdown]
## 改进的约束感知优化更新方法
## =====================================================================================================

import torch
import torch.nn.functional as F
from scipy.optimize import linprog
import numpy as np

def extract_adjacency_probabilities(x0_pred):
    """从x0_pred中提取邻接矩阵概率"""
    # x0_pred shape: (batch_size, 2, num_nodes, num_nodes)
    x0_pred_clamped = torch.clamp(x0_pred, min=-50.0, max=50.0)
    x0_pred_prob = x0_pred_clamped.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
    adj_prob_matrix = x0_pred_prob[:, :, :, 1]  # 取边存在的概率
    return adj_prob_matrix

def calculate_constrained_update_pytorch(x0_pred, gradient, budget=1.0, rank_penalty=100.0, 
                                       norm_penalty=50.0, epsilon=1e-4, batch_idx=0):
    """
    计算约束感知的更新方向（PyTorch集成版本）
    
    Args:
        x0_pred: torch.Tensor - 当前预测 (batch_size, 2, num_nodes, num_nodes)
        gradient: torch.Tensor - 梯度 (batch_size, 2, num_nodes, num_nodes)
        budget: float - L1范数更新预算
        rank_penalty: float - 排序翻转惩罚权重
        norm_penalty: float - 归一化约束惩罚权重
        epsilon: float - 约束松弛参数
        batch_idx: int - 处理的批次索引
        
    Returns:
        torch.Tensor - 更新后的x0_pred
    """
    device = x0_pred.device
    batch_size = x0_pred.shape[0]
    
    # 提取当前批次的数据
    x0_pred_single = x0_pred[batch_idx]  # (2, num_nodes, num_nodes)
    grad_single = gradient[batch_idx]    # (2, num_nodes, num_nodes)
    
    # 提取邻接矩阵概率 (只取边存在的概率)
    adj_prob = extract_adjacency_probabilities(x0_pred_single.unsqueeze(0))[0]  # (num_nodes, num_nodes)
    grad_adj = grad_single[1]  # 只处理边存在的梯度
    
    # 转换为numpy进行优化
    C = adj_prob.detach().cpu().numpy()
    g = grad_adj.detach().cpu().numpy()
    
    # 调用线性规划优化
    delta, slacks_r, slacks_n = calculate_fully_constrained_update(
        C, g, budget, rank_penalty, norm_penalty, epsilon
    )
    
    if delta is not None:
        # 转换回PyTorch tensor
        delta_tensor = torch.from_numpy(delta).float().to(device)
        
        # 构造完整的更新
        x0_pred_updated = x0_pred.clone()
        
        # 更新边存在的logit
        adj_prob_updated = adj_prob + delta_tensor
        adj_prob_updated = torch.clamp(adj_prob_updated, min=1e-8, max=1.0-1e-8)
        
        # 转换回logit形式
        x0_pred_updated[batch_idx, 1] = torch.log(adj_prob_updated + 1e-8)
        x0_pred_updated[batch_idx, 0] = torch.log(1 - adj_prob_updated + 1e-8)
        
        return x0_pred_updated
    else:
        print("线性规划求解失败，使用原始梯度更新")
        return x0_pred - 0.01 * gradient

# 移除apply_symmetry_constraint函数，因为VRP问题不需要对称性约束

def adaptive_budget_scheduler(step, total_steps, initial_budget=1.0, final_budget=0.1):
    """自适应预算调度器"""
    progress = step / total_steps
    # 指数衰减
    budget = initial_budget * (final_budget / initial_budget) ** progress
    return budget

def constrained_optimizer_step(x0_pred, gradient, step, total_steps, **kwargs):
    """约束感知的优化器步骤""" 
    # 自适应预算
    # budget = adaptive_budget_scheduler(step, total_steps)
    budget = 1.0
    
    # 计算约束感知的更新
    return calculate_constrained_update_pytorch(
        x0_pred, gradient, budget=budget, **kwargs
    )

# %% [markdown]
## 混合优化策略
## =====================================================================================================
from torch.optim.optimizer import Optimizer
class HybridOptimizer(Optimizer):
    """约束感知优化器：直接使用约束感知的更新方向"""
    
    def __init__(self, params, lr=0.01, rank_penalty=100.0, norm_penalty=50.0, **kwargs):
        # 设置默认参数
        defaults = dict(lr=lr, rank_penalty=rank_penalty, norm_penalty=norm_penalty, **kwargs)
        super().__init__(params, defaults)
        
        # 初始化优化器状态
        self.step_count = 0
        
        # 保存约束相关参数，方便访问
        self.constraint_kwargs = {
            'rank_penalty': rank_penalty,
            'norm_penalty': norm_penalty,
            **kwargs
        }
        
    def step(self, total_steps=None):
        """执行一步优化"""
        self.step_count += 1
        
        for group in self.param_groups:
            rank_penalty = group['rank_penalty']
            norm_penalty = group['norm_penalty']
            
            for param in group['params']:
                if param.grad is not None:
                    with torch.no_grad():
                        # 使用约束感知更新，传递从group中获取的参数
                        param.data = constrained_optimizer_step(
                            param, param.grad, self.step_count, total_steps,
                            rank_penalty=rank_penalty,
                            norm_penalty=norm_penalty,
                            **{k: v for k, v in group.items() if k not in ['params', 'lr', 'rank_penalty', 'norm_penalty']}
                        )
                        
    def get_constraint_metrics(self):
        """获取约束相关指标"""
        metrics = {}
        param_idx = 0
        
        for group in self.param_groups:
            for param in group['params']:
                adj_prob = extract_adjacency_probabilities(param.unsqueeze(0) if param.dim() == 3 else param)[0]
                
                # 行和偏差
                row_sums = adj_prob.sum(dim=1)
                row_deviation = torch.mean(torch.abs(row_sums - 1.0)).item()
                
                metrics[f'param_{param_idx}_row_deviation'] = row_deviation
                param_idx += 1
                
        return metrics
    
    @property
    def lr(self):
        """获取学习率（从第一个参数组）"""
        return self.param_groups[0]['lr']
    
    @property
    def constraint_frequency(self):
        """约束更新频率（默认每步都更新）"""
        return 1

# %% [markdown]
## 使用混合优化器的改进训练循环
## =====================================================================================================
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

model = TestTSPModel()
model.rl_debug = False
 
print("\n" + "="*80)
print("使用约束感知优化器的改进训练循环")
print("="*80)

# 重新初始化参数
x0_pred_constrained = torch.nn.Parameter(adj_matrix_onehot.clone().detach())

# 创建约束感知优化器
hybrid_optimizer = HybridOptimizer(
    params=[x0_pred_constrained],
    lr=0.02, 
    rank_penalty=50.0,
    norm_penalty=25.0
)

# 训练设置
num_iters_constrained = 200
log_constrained = defaultdict(list)

print(f"开始约束感知 RL 微调（{num_iters_constrained} 步）")
print(f"约束更新频率: 每{hybrid_optimizer.constraint_frequency}步")
print(f"学习率: {hybrid_optimizer.lr}")
print(f"排序惩罚权重: {hybrid_optimizer.constraint_kwargs.get('rank_penalty', 0)}")
print(f"归一化惩罚权重: {hybrid_optimizer.constraint_kwargs.get('norm_penalty', 0)}")

# 训练循环
pbar_constrained = tqdm(range(num_iters_constrained), desc="约束感知优化")
for step in pbar_constrained:
    hybrid_optimizer.zero_grad()
    
    # 计算损失
    rl_loss, metrics = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred_constrained,
        points=points,
        gt_tour=gt_tour,
        current_problem_type=problem_type,
        batch_idx=batch_idx
    )
    
    # 反向传播
    rl_loss.backward()
    
    # 使用混合优化器更新
    # print('更新参数')
    hybrid_optimizer.step(total_steps=num_iters_constrained)
    
    # 获取约束指标
    constraint_metrics = hybrid_optimizer.get_constraint_metrics()
    
    # 记录日志
    log_constrained["step"].append(step)
    log_constrained["loss"].append(rl_loss.item())
    for k, v in metrics.items():
        log_constrained[k].append(v)
    for k, v in constraint_metrics.items():
        log_constrained[k].append(v)
    
    # 更新进度条
    pbar_constrained.set_postfix({
        'loss': f'{rl_loss.item():.4f}',
        'avg_cost': f'{metrics["avg_pred_cost"]:.4f}',
        'best_cost': f'{metrics["best_pred_cost"]:.4f}',
        'row_dev': f'{constraint_metrics.get("param_0_row_deviation", 0):.3f}'
    })

print(f"\n约束感知微调完成！")

# 保存结果
x0_pred_constrained_final = x0_pred_constrained.detach()
df_log_constrained = pd.DataFrame(log_constrained)

# %% [markdown]
## 对比分析：标准优化 vs 约束感知优化
## =====================================================================================================

print("\n" + "="*50)
print("对比分析：标准优化 vs 约束感知优化")
print("="*50)

# 使用相同的温度进行最终评估
model.pomo_temperature = 0.0

# 评估标准优化结果
with torch.no_grad():
    standard_loss, standard_metrics = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred.detach(),
        points=points,
        gt_tour=gt_tour,
        current_problem_type=problem_type,
        batch_idx=batch_idx
    )

# 评估约束感知优化结果
with torch.no_grad():
    constrained_loss, constrained_metrics = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred_constrained_final,
        points=points,
        gt_tour=gt_tour,
        current_problem_type=problem_type,
        batch_idx=batch_idx
    )

# 性能对比
print(f"\n性能对比:")
print(f"  标准优化:")
print(f"    - 最终损失: {standard_loss.item():.4f}")
print(f"    - 平均成本: {standard_metrics['avg_pred_cost']:.4f}")
print(f"    - 最佳成本: {standard_metrics['best_pred_cost']:.4f}")
print(f"    - 成本差距: {standard_metrics['cost_gap_percent']:.2f}%")

print(f"  约束感知优化:")
print(f"    - 最终损失: {constrained_loss.item():.4f}")
print(f"    - 平均成本: {constrained_metrics['avg_pred_cost']:.4f}")
print(f"    - 最佳成本: {constrained_metrics['best_pred_cost']:.4f}")
print(f"    - 成本差距: {constrained_metrics['cost_gap_percent']:.2f}%")

# 改进计算
cost_improvement = (standard_metrics['best_pred_cost'] - constrained_metrics['best_pred_cost']) / standard_metrics['best_pred_cost'] * 100

print(f"\n改进情况:")
print(f"  成本改进: {cost_improvement:.2f}%")
# %% [markdown]
## 可视化对比结果
## =====================================================================================================

def plot_training_comparison():
    """Plot training comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss comparison
    axes[0,0].plot(df_log['step'], df_log['loss'], 'b-', label='Standard Opt', alpha=0.7)
    axes[0,0].plot(df_log_constrained['step'], df_log_constrained['loss'], 'r-', label='Constrained Opt', alpha=0.7)
    axes[0,0].set_title('Loss Comparison')
    axes[0,0].set_xlabel('Iterations')
    axes[0,0].set_ylabel('Loss Value')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Cost comparison
    axes[0,1].plot(df_log['step'], df_log['best_pred_cost'], 'b-', label='Standard Opt', alpha=0.7)
    axes[0,1].plot(df_log_constrained['step'], df_log_constrained['best_pred_cost'], 'r-', label='Constrained Opt', alpha=0.7)
    axes[0,1].axhline(y=df_log['avg_gt_cost'].mean(), color='g', linestyle='--', label='Optimal Cost')
    axes[0,1].set_title('Best Cost Comparison')
    axes[0,1].set_xlabel('Iterations')
    axes[0,1].set_ylabel('Path Cost')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Constraint violation comparison
    if 'param_0_row_deviation' in df_log_constrained.columns:
        axes[0,2].plot(df_log_constrained['step'], df_log_constrained['param_0_row_deviation'], 'r-', label='Row Sum Violation')
        axes[0,2].set_title('Row Sum Constraint Violation')
        axes[0,2].set_xlabel('Iterations')
        axes[0,2].set_ylabel('Violation Degree')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
    
    # Adjacency matrix visualization
    standard_adj = extract_adjacency_probabilities(x0_pred_optim)[0]
    constrained_adj = extract_adjacency_probabilities(x0_pred_constrained_final)[0]
    
    im1 = axes[1,0].imshow(standard_adj.detach().cpu().numpy(), cmap='YlOrRd', vmin=0, vmax=1)
    axes[1,0].set_title('Standard Opt - Adjacency Matrix')
    plt.colorbar(im1, ax=axes[1,0])
    
    im2 = axes[1,1].imshow(constrained_adj.detach().cpu().numpy(), cmap='YlOrRd', vmin=0, vmax=1)
    axes[1,1].set_title('Constrained Opt - Adjacency Matrix')
    plt.colorbar(im2, ax=axes[1,1])
    
    # Row sum constraint violation heatmap
    row_violations = torch.abs(constrained_adj.sum(dim=1) - 1.0).unsqueeze(1).expand(-1, constrained_adj.shape[1])
    im3 = axes[1,2].imshow(row_violations.detach().cpu().numpy(), cmap='Reds', vmin=0, vmax=0.1)
    axes[1,2].set_title('Row Sum Constraint Violation Heatmap')
    plt.colorbar(im3, ax=axes[1,2])
    
    plt.tight_layout()
    plt.show()

plot_training_comparison()


# %%

# 创建3x3子图布局
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 获取第一个样本的数据
sample_idx = 0
standard_adj = extract_adjacency_probabilities(x0_pred_optim)[sample_idx]
constrained_adj = extract_adjacency_probabilities(x0_pred_constrained_final)[sample_idx]
base_adj = extract_adjacency_probabilities(x0_pred)[sample_idx]
sample_points = points[sample_idx]
gt_tour_sample = gt_tour[sample_idx].detach().cpu().numpy()

# 绘制基线模型结果
plot_adj_matrix_and_route(base_adj, sample_points, 
                         "Baseline", axes[0,0], axes[0,1])

# 绘制标准优化结果
plot_adj_matrix_and_route(standard_adj, sample_points,
                         "Standard Opt", axes[1,0], axes[1,1])

# 绘制约束优化结果
plot_adj_matrix_and_route(constrained_adj, sample_points,
                         "Constrained Opt", axes[2,0], axes[2,1])

# 绘制Ground Truth最优解
plot_ground_truth_route(gt_tour_sample, sample_points,
                       "Ground Truth", axes[0,2], axes[1,2])

# 在第三行第三列显示性能对比
ax_comparison = axes[2,2]
methods = ['Baseline', 'Standard Opt', 'Constrained Opt', 'Ground Truth']

# 计算基线成本（使用微调前的结果）
with torch.no_grad():
    base_loss, base_metrics = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred.detach(),
        points=points,
        gt_tour=gt_tour,
        current_problem_type=problem_type,
        batch_idx=batch_idx
    )

costs = [
    base_metrics['best_pred_cost'],
    standard_metrics['best_pred_cost'],
    constrained_metrics['best_pred_cost'],
    gt_cost
]

colors = ['blue', 'orange', 'green', 'red']
bars = ax_comparison.bar(methods, costs, color=colors, alpha=0.7)
ax_comparison.set_title('Cost Comparison')
ax_comparison.set_ylabel('Path Cost')
ax_comparison.tick_params(axis='x', rotation=45)

# 在柱状图上显示数值
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax_comparison.text(bar.get_x() + bar.get_width()/2., height,
                      f'{cost:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%

# %% [markdown]
## 5节点VRP测试案例 - 完整的对比和可视化
## =====================================================================================================

def create_simple_vrp_test():
    """创建一个简单的5节点VRP测试问题,只考虑容量约束"""
    
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建5个节点的VRP问题
    num_nodes = 5
    batch_size = 1
    
    # 手工设计的节点位置，确保问题有清晰的结构
    coordinates = np.array([
        [0.5, 0.5],   # depot (中心)
        [0.2, 0.2],   # 客户1 (左下区域)
        [0.3, 0.1],   # 客户2 (左下区域)
        [0.1, 0.3],   # 客户3 (左下区域) 
        [0.8, 0.8]    # 客户4 (右上区域)
    ])
    
    # 创建7维特征向量,除了坐标和需求量外,其他属性都设为0
    points_with_features = np.zeros((num_nodes, 7))
    points_with_features[:, :2] = coordinates  # 坐标
    
    # 只设置需求量,其他约束相关的属性都为0
    demands = np.array([0.0, 0.3, 0.3, 0.3, 0.3])  # depot需求为0,客户需求设为0.3
    points_with_features[:, 2] = demands
    
    # 转换为torch tensor
    points_tensor = torch.from_numpy(points_with_features).float().unsqueeze(0)  # [1, 5, 7]
    # 创建初始的邻接矩阵 (随机初始化)
    x0_pred_init = torch.randn(1, 2, num_nodes, num_nodes) * 0.5
    
    print(f"创建5节点VRP测试问题(仅考虑容量约束):")
    print(f"  - 节点坐标:\n{coordinates}")
    print(f"  - 需求量: {demands}") 
    
    return points_tensor, x0_pred_init, points_with_features

def solve_vrp_with_ortools_simple(points_with_features):
    """使用OR-Tools求解简单VRP问题"""     
    # 提取CVRP数据
    cvrp_data = extract_cvrp_data(points_with_features, "VRP")
    
    # 求解
    solver_result = solve_cvrp_with_ortools(cvrp_data, time_limit_seconds=30)
    
    if solver_result and solver_result.get('success'):
        print(f"✅ OR-Tools求解成功!")
        print(f"  - 使用车辆数: {solver_result['num_vehicles_used']}")
        print(f"  - 总距离: {solver_result['total_distance']}")
        print(f"  - 路径:")
        for i, route in enumerate(solver_result['routes']):
            print(f"    车辆{i}: {' -> '.join(map(str, route))}")
        
        return solver_result
    else:
        print("❌ OR-Tools求解失败")
        return None

def visualize_optimization_step(step, x0_pred, points, gt_tour, ortools_result, 
                              adj_prob, route_info, fig_size=(20, 12)):
    """可视化单步优化结果 - 显示当前步骤和最优解的热力图与路径对比"""
    
    fig, axes = plt.subplots(3, 4, figsize=fig_size)
    
    coordinates = points[0, :, :2].detach().cpu().numpy()
    num_nodes = len(coordinates)
    
    # === 第一行：当前步骤的结果 ===
    
    # 1. 当前步骤的邻接矩阵热力图
    adj_prob_np = adj_prob.detach().cpu().numpy()
    im1 = axes[0,0].imshow(adj_prob_np, cmap='YlOrRd', vmin=0, vmax=1)
    axes[0,0].set_title(f'Step {step}: Current Adjacency Matrix')
    axes[0,0].set_xlabel('To Node')
    axes[0,0].set_ylabel('From Node')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 2. 当前步骤基于热力图生成的路径图
    axes[0,1].scatter(coordinates[:, 0], coordinates[:, 1], s=200, c='red', alpha=0.7, zorder=5)
    axes[0,1].scatter(coordinates[0, 0], coordinates[0, 1], s=300, c='blue', marker='s', alpha=0.9, zorder=6, label='Depot')
    
    # 添加节点标签
    for i in range(len(coordinates)):
        axes[0,1].annotate(f'{i}', (coordinates[i, 0], coordinates[i, 1]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    
    # 绘制基于邻接矩阵的连接（使用不同透明度表示概率）
    threshold = 0.2  # 降低阈值以显示更多连接
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            if adj_prob_np[i,j] > threshold:
                alpha = float(adj_prob_np[i,j])
                linewidth = 1 + 3 * alpha  # 线宽随概率变化
                axes[0,1].plot([coordinates[i,0], coordinates[j,0]], 
                             [coordinates[i,1], coordinates[j,1]], 
                             'g-', alpha=alpha, linewidth=linewidth)
    
    axes[0,1].set_title(f'Step {step}: Current Graph Structure')
    axes[0,1].set_xlim(-0.1, 1.1)
    axes[0,1].set_ylim(-0.1, 1.1)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # 3. 行和约束违反情况
    row_sums = adj_prob.sum(dim=1)
    violations = torch.abs(row_sums - 1.0)
    
    axes[0,2].bar(range(len(violations)), violations.detach().cpu().numpy(), alpha=0.7, color='red')
    axes[0,2].axhline(y=0, color='green', linestyle='--', alpha=0.7)
    axes[0,2].set_title(f'Step {step}: Row Sum Violations')
    axes[0,2].set_xlabel('Node')
    axes[0,2].set_ylabel('|Row Sum - 1|')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. 当前步骤的成本信息
    if route_info:
        current_cost = route_info.get('current_cost', 0)
        axes[0,3].text(0.5, 0.7, f'Current Cost: {current_cost:.3f}', 
                      ha='center', va='center', transform=axes[0,3].transAxes, fontsize=14, fontweight='bold')
        
        # 显示行和统计
        row_sums_np = row_sums.detach().cpu().numpy()
        axes[0,3].text(0.5, 0.5, f'Row Sums: {[f"{x:.2f}" for x in row_sums_np]}', 
                      ha='center', va='center', transform=axes[0,3].transAxes, fontsize=10)
        axes[0,3].text(0.5, 0.3, f'Mean Violation: {violations.mean().item():.3f}', 
                      ha='center', va='center', transform=axes[0,3].transAxes, fontsize=12)
    
    axes[0,3].set_title(f'Step {step}: Current Status')
    axes[0,3].axis('off')
    
    # === 第二行：OR-Tools最优解 ===
    
    if ortools_result and ortools_result.get('routes'):
        # 1. 构建OR-Tools解的邻接矩阵
        ortools_adj_matrix = np.zeros((num_nodes, num_nodes))
        for route in ortools_result['routes']:
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                ortools_adj_matrix[from_node, to_node] = 1.0
        
        # 2. OR-Tools解的邻接矩阵热力图
        im2 = axes[1,0].imshow(ortools_adj_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        axes[1,0].set_title('OR-Tools: Optimal Adjacency Matrix')
        axes[1,0].set_xlabel('To Node')
        axes[1,0].set_ylabel('From Node')
        plt.colorbar(im2, ax=axes[1,0])
        
        # 3. OR-Tools解的路径图
        axes[1,1].scatter(coordinates[:, 0], coordinates[:, 1], s=200, c='red', alpha=0.7, zorder=5)
        axes[1,1].scatter(coordinates[0, 0], coordinates[0, 1], s=300, c='blue', marker='s', alpha=0.9, zorder=6, label='Depot')
        
        # 添加节点标签
        for i in range(len(coordinates)):
            axes[1,1].annotate(f'{i}', (coordinates[i, 0], coordinates[i, 1]), 
                             xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
        
        # 绘制OR-Tools路径
        colors = ['purple', 'orange', 'brown', 'pink', 'cyan']
        for route_idx, route in enumerate(ortools_result['routes']):
            color = colors[route_idx % len(colors)]
            for i in range(len(route) - 1):
                start = coordinates[route[i]]
                end = coordinates[route[i + 1]]
                axes[1,1].plot([start[0], end[0]], [start[1], end[1]], 
                             color=color, linewidth=3, alpha=0.8, 
                             label=f'Vehicle {route_idx}' if i == 0 else '')
        
        axes[1,1].set_title('OR-Tools: Optimal Route')
        axes[1,1].set_xlim(-0.1, 1.1)
        axes[1,1].set_ylim(-0.1, 1.1)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        # 4. OR-Tools解的行和检查
        ortools_row_sums = np.sum(ortools_adj_matrix, axis=1)
        ortools_violations = np.abs(ortools_row_sums - 1.0)
        
        axes[1,2].bar(range(len(ortools_violations)), ortools_violations, alpha=0.7, color='purple')
        axes[1,2].axhline(y=0, color='green', linestyle='--', alpha=0.7)
        axes[1,2].set_title('OR-Tools: Row Sum Violations')
        axes[1,2].set_xlabel('Node')
        axes[1,2].set_ylabel('|Row Sum - 1|')
        axes[1,2].grid(True, alpha=0.3)
        
        # 5. OR-Tools解的成本信息
        ortools_cost = route_info.get('ortools_cost', 0) if route_info else 0
        axes[1,3].text(0.5, 0.7, f'OR-Tools Cost: {ortools_cost:.3f}', 
                      ha='center', va='center', transform=axes[1,3].transAxes, fontsize=14, fontweight='bold')
        axes[1,3].text(0.5, 0.5, f'Vehicles Used: {ortools_result["num_vehicles_used"]}', 
                      ha='center', va='center', transform=axes[1,3].transAxes, fontsize=12)
        axes[1,3].text(0.5, 0.3, f'Mean Violation: {ortools_violations.mean():.3f}', 
                      ha='center', va='center', transform=axes[1,3].transAxes, fontsize=12)
        
        axes[1,3].set_title('OR-Tools: Status')
        axes[1,3].axis('off')
    else:
        # 如果没有OR-Tools结果，显示提示信息
        for i in range(4):
            axes[1,i].text(0.5, 0.5, 'OR-Tools Solution\nNot Available', 
                          ha='center', va='center', transform=axes[1,i].transAxes, fontsize=14)
            axes[1,i].set_title('OR-Tools: N/A')
            axes[1,i].axis('off')
    
    # === 第三行：Ground Truth解 ===
    
    # 1. 构建Ground Truth解的邻接矩阵
    gt_adj_matrix = np.zeros((num_nodes, num_nodes))
    gt_path = gt_tour[0].detach().cpu().numpy()
    for i in range(len(gt_path) - 1):
        from_node = gt_path[i]
        to_node = gt_path[i + 1]
        gt_adj_matrix[from_node, to_node] = 1.0
    
    # 2. Ground Truth解的邻接矩阵热力图
    im3 = axes[2,0].imshow(gt_adj_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    axes[2,0].set_title('Ground Truth: Adjacency Matrix')
    axes[2,0].set_xlabel('To Node')
    axes[2,0].set_ylabel('From Node')
    plt.colorbar(im3, ax=axes[2,0])
    
    # 3. Ground Truth解的路径图
    axes[2,1].scatter(coordinates[:, 0], coordinates[:, 1], s=200, c='red', alpha=0.7, zorder=5)
    axes[2,1].scatter(coordinates[0, 0], coordinates[0, 1], s=300, c='blue', marker='s', alpha=0.9, zorder=6, label='Depot')
    
    # 添加节点标签
    for i in range(len(coordinates)):
        axes[2,1].annotate(f'{i}', (coordinates[i, 0], coordinates[i, 1]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    
    # 绘制Ground Truth路径
    for i in range(len(gt_path) - 1):
        start = coordinates[gt_path[i]]
        end = coordinates[gt_path[i + 1]]
        axes[2,1].plot([start[0], end[0]], [start[1], end[1]], 
                     'red', linewidth=3, alpha=0.8)
    
    axes[2,1].set_title('Ground Truth: Route')
    axes[2,1].set_xlim(-0.1, 1.1)
    axes[2,1].set_ylim(-0.1, 1.1)
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].legend()
    
    # 4. Ground Truth解的行和检查
    gt_row_sums = np.sum(gt_adj_matrix, axis=1)
    gt_violations = np.abs(gt_row_sums - 1.0)
    
    axes[2,2].bar(range(len(gt_violations)), gt_violations, alpha=0.7, color='red')
    axes[2,2].axhline(y=0, color='green', linestyle='--', alpha=0.7)
    axes[2,2].set_title('Ground Truth: Row Sum Violations')
    axes[2,2].set_xlabel('Node')
    axes[2,2].set_ylabel('|Row Sum - 1|')
    axes[2,2].grid(True, alpha=0.3)
    
    # 5. Ground Truth解的成本信息
    gt_cost = route_info.get('gt_cost', 0) if route_info else 0
    axes[2,3].text(0.5, 0.7, f'Ground Truth Cost: {gt_cost:.3f}', 
                  ha='center', va='center', transform=axes[2,3].transAxes, fontsize=14, fontweight='bold')
    axes[2,3].text(0.5, 0.5, f'Path: {gt_path.tolist()}', 
                  ha='center', va='center', transform=axes[2,3].transAxes, fontsize=10)
    axes[2,3].text(0.5, 0.3, f'Mean Violation: {gt_violations.mean():.3f}', 
                  ha='center', va='center', transform=axes[2,3].transAxes, fontsize=12)
    
    axes[2,3].set_title('Ground Truth: Status')
    axes[2,3].axis('off')
    
    # === 总体对比 ===
    
    # 添加总标题
    fig.suptitle(f'Step {step} Optimization Progress: Current vs Optimal Solutions', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""带可视化的约束感知优化"""

print("\n" + "="*80)
print("5节点VRP问题 - 约束感知优化 + 逐步可视化")
print("="*80)

# 1. 创建测试数据
points, x0_pred_init, points_with_features = create_simple_vrp_test()

# 2. 使用OR-Tools求解最优解
ortools_result = solve_vrp_with_ortools_simple(points_with_features)
print(ortools_result)
# %% 
# 3. 创建ground truth tour
if ortools_result and ortools_result.get('success'):
    # 从OR-Tools结果构建ground truth tour
    gt_routes = ortools_result['routes']
    gt_tour = []
    for route in gt_routes:
        gt_tour.extend(route)
    gt_tour = torch.tensor([gt_tour]).long()  # [1, num_nodes]
else:
    print("警告: 使用随机生成的ground truth tour")
    gt_tour = torch.randint(0, points.shape[1], (1, points.shape[1]))

print("\n最优解验证:")
print(f"Ground truth tour: {gt_tour.tolist()}")

# 可视化初始状态
print("\n初始状态可视化:")
visualize_optimization_step(
    step=0,
    x0_pred=x0_pred_init,
    points=points,
    gt_tour=gt_tour,
    ortools_result=ortools_result,
    adj_prob=extract_adjacency_probabilities(x0_pred_init)[0],
    route_info=None
)

# %%
# 3. 设置优化参数
model = TestTSPModel()
model.rl_debug = False
model.pomo_temperature = 0.1

# 创建可学习参数
x0_pred_param = torch.nn.Parameter(x0_pred_init.clone())

# 创建约束感知优化器
optimizer = HybridOptimizer(
    params=[x0_pred_param],
    lr=0.05,
    rank_penalty=100.0,
    norm_penalty=0.01
)

# 4. 逐步优化并可视化
num_steps = 50
visualization_steps = [0, 5, 10, 15, 19, 25, 30, 35, 40, 45, 49]  # 选择特定步骤进行可视化

print(f"\n开始逐步优化（共{num_steps}步）...")

for step in range(num_steps):
    optimizer.zero_grad()
    
    # 计算损失
    rl_loss, metrics = model.compute_reinforcement_learning_loss(
        x0_pred=x0_pred_param,
        points=points,
        gt_tour=gt_tour,
        current_problem_type="VRP",
        batch_idx=0
    )
    
    # 反向传播
    rl_loss.backward()
    
    # 约束感知更新
    optimizer.step(total_steps=num_steps)
    
    # 获取当前邻接矩阵
    with torch.no_grad():
        adj_prob = extract_adjacency_probabilities(x0_pred_param)[0]
    
    # 计算路径信息
    route_info = {
        'current_cost': metrics['best_pred_cost'],
        'ortools_cost': ortools_result['total_distance']/1000 if ortools_result else 0,
        'gt_cost': metrics['avg_gt_cost']
    }
    
    # 打印进度
    constraint_metrics = optimizer.get_constraint_metrics()
    row_violation = constraint_metrics.get('param_0_row_deviation', 0)
    
    print(f"Step {step:2d}: Loss={rl_loss.item():.4f}, Cost={metrics['best_pred_cost']:.4f}, "
            f"Row Violation={row_violation:.4f}")
    
    # 可视化特定步骤
    if step in visualization_steps:
        print(f"\n=== 可视化步骤 {step} ===")
        visualize_optimization_step(
            step, x0_pred_param, points, gt_tour, ortools_result, 
            adj_prob, route_info
        )

print(f"\n✅ 优化完成!")

# 5. 最终结果总结
print(f"\n最终结果总结:")
print(f"  - 最终损失: {rl_loss.item():.4f}")
print(f"  - 最终成本: {metrics['best_pred_cost']:.4f}")
print(f"  - OR-Tools成本: {ortools_result['total_distance']/1000 if ortools_result else 'N/A'}")
print(f"  - Ground Truth成本: {metrics['avg_gt_cost']:.4f}")

with torch.no_grad():
    final_adj_prob = extract_adjacency_probabilities(x0_pred_param)[0]
    row_sums = final_adj_prob.sum(dim=1)
    print(f"  - 最终行和: {row_sums.tolist()}")
    print(f"  - 行和违反度: {torch.abs(row_sums - 1.0).mean().item():.4f}")
     
# %%
