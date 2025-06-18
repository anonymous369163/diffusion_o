"""Lightning module for training the DIFUSCO TSP model."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch  # , merge_tours  # 注释掉因为改用强化学习方法


def greedy_tsp_solver_batch_pomo(adj_matrix_batch, temperature=1.0):
    """
    POMO版本的批量TSP求解器，同时从所有节点作为起始点进行求解，支持探索
    Args:
        adj_matrix_batch: torch.Tensor of shape (batch_size, num_nodes, num_nodes) - 邻接矩阵批次
        temperature: float - 控制探索程度的温度参数，越大越随机，越小越贪婪 (默认1.0)
    Returns:
        tours: torch.Tensor of shape (batch_size * num_nodes, num_nodes + 1) - 所有路径
        log_probs: torch.Tensor of shape (batch_size * num_nodes,) - 每个路径的对数概率
    """
    batch_size, num_nodes, _ = adj_matrix_batch.shape
    device = adj_matrix_batch.device
    
    # 扩展维度：为每个样本的每个节点作为起始点创建副本
    # 从 (batch_size, num_nodes, num_nodes) 扩展为 (batch_size * num_nodes, num_nodes, num_nodes)
    expanded_adj = adj_matrix_batch.unsqueeze(1).expand(-1, num_nodes, -1, -1)
    expanded_adj = expanded_adj.reshape(batch_size * num_nodes, num_nodes, num_nodes)
    
    # 创建起始节点索引：[0,1,2,...,num_nodes-1, 0,1,2,...,num_nodes-1, ...]
    start_nodes = torch.arange(num_nodes, device=device).repeat(batch_size)  # shape: (batch_size * num_nodes,)
    
    # 初始化路径
    tours = torch.zeros(batch_size * num_nodes, num_nodes + 1, dtype=torch.long, device=device)
    tours[:, 0] = start_nodes  # 设置起始节点
    
    # 当前节点位置
    current_nodes = start_nodes.clone()
    
    # 累积对数概率
    log_probs = torch.zeros(batch_size * num_nodes, device=device)
    
    # 初始化访问掩码（使用非原地操作的方式）
    visited_mask = torch.zeros(batch_size * num_nodes, num_nodes, dtype=torch.bool, device=device)
    # 使用非原地方式标记起始节点
    start_mask = torch.zeros_like(visited_mask)
    start_mask.scatter_(1, start_nodes.unsqueeze(1), True)
    visited_mask = visited_mask | start_mask
    
    # 迭代构建路径
    for step in range(num_nodes - 1):
        # 获取当前节点到所有节点的边权重
        batch_indices = torch.arange(batch_size * num_nodes, device=device)
        current_edges = expanded_adj[batch_indices, current_nodes]  # shape: (batch_size * num_nodes, num_nodes)
        
        # 应用访问掩码：已访问节点设为负无穷
        masked_edges = torch.where(visited_mask, 
                                 torch.tensor(-float('inf'), device=device), 
                                 current_edges)
        
        # 根据温度参数控制探索程度
        if temperature <= 0.0:
            # 温度为0或负数时使用纯贪婪选择
            edge_probs = F.softmax(masked_edges, dim=-1)  # shape: (batch_size * num_nodes, num_nodes)
            next_nodes = torch.argmax(masked_edges, dim=-1)  # shape: (batch_size * num_nodes,)
        else:
            # 使用温度参数调整logits以控制探索程度
            scaled_logits = masked_edges / temperature
            
            # 计算概率分布
            edge_probs = F.softmax(scaled_logits, dim=-1)  # shape: (batch_size * num_nodes, num_nodes)
            
            # 根据概率分布进行多项式采样，引入探索
            next_nodes = torch.multinomial(edge_probs, num_samples=1).squeeze(-1)  # shape: (batch_size * num_nodes,)
        
        # 记录选择的对数概率
        selected_probs = edge_probs.gather(1, next_nodes.unsqueeze(1)).squeeze(1)
        log_probs += torch.log(selected_probs + 1e-8)
        
        # 更新路径
        tours[:, step + 1] = next_nodes
        
        # 使用非原地操作更新访问掩码
        next_mask = torch.zeros_like(visited_mask)
        next_mask.scatter_(1, next_nodes.unsqueeze(1), True)
        visited_mask = visited_mask | next_mask
        
        # 更新当前节点
        current_nodes = next_nodes
    
    # 添加回到起始节点的路径
    tours[:, -1] = start_nodes
    
    # 计算回到起始节点的对数概率
    return_edges = expanded_adj[batch_indices, current_nodes, start_nodes]
    log_probs += torch.log(return_edges + 1e-8)
    
    return tours, log_probs


def greedy_tsp_solver_batch(adj_matrix_batch, start_node=0):
    """
    原始版本的批量贪婪TSP求解器（保持兼容性）
    Args:
        adj_matrix_batch: torch.Tensor of shape (batch_size, num_nodes, num_nodes) - 邻接矩阵批次
        start_node: int - 起始节点索引
    Returns:
        tours: list of lists - 每个样本的TSP路径
        log_probs: torch.Tensor - 每个路径的对数概率
    """
    batch_size, num_nodes, _ = adj_matrix_batch.shape
    device = adj_matrix_batch.device
    tours = []
    log_probs = []
    
    # 为每个样本求解TSP
    for b in range(batch_size):
        tour = [start_node]
        current_node = start_node
        tour_log_prob = 0.0
        
        # 初始化访问掩码
        visited_indices = [start_node]
        
        # 贪婪选择下一个节点
        for step in range(num_nodes - 1):
            # 创建当前步骤的访问掩码（避免原地操作）
            current_visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            for idx in visited_indices:
                current_visited[idx] = True
            
            # 获取当前节点到所有节点的原始概率
            edge_logits = adj_matrix_batch[b][current_node].clone()  # 保持梯度连接
            
            # 创建掩码版本的logits（避免原地操作）
            masked_logits = torch.where(current_visited, 
                                      torch.tensor(-float('inf'), device=device), 
                                      edge_logits)
            
            # 使用softmax获取概率分布
            edge_probs = F.softmax(masked_logits, dim=-1)
            
            # 贪婪选择概率最大的节点
            next_node = torch.argmax(edge_probs).item()
            
            # 记录选择的对数概率
            tour_log_prob += torch.log(edge_probs[next_node] + 1e-8)
            
            # 更新状态
            tour.append(next_node)
            visited_indices.append(next_node)
            current_node = next_node
        
        # 回到起始节点
        tour.append(start_node) 
        return_edge_prob = adj_matrix_batch[b][current_node, start_node]  # 保持梯度连接
        tour_log_prob += torch.log(return_edge_prob + 1e-8)
        
        tours.append(tour)
        log_probs.append(tour_log_prob)
    
    return tours, torch.stack(log_probs)


def calculate_euclidean_distance_batch(points_batch):
    """
    批量计算欧几里得距离矩阵
    Args:
        points_batch: torch.Tensor of shape (batch_size, num_nodes, 2)
    Returns:
        distance_matrices: torch.Tensor of shape (batch_size, num_nodes, num_nodes)
    """
    batch_size, num_nodes, coord_dim = points_batch.shape
    device = points_batch.device
    
    # 使用广播计算距离矩阵
    points_expanded_i = points_batch.unsqueeze(2)  # (batch_size, num_nodes, 1, 2)
    points_expanded_j = points_batch.unsqueeze(1)  # (batch_size, 1, num_nodes, 2)
    
    # 计算欧几里得距离
    distance_matrices = torch.sqrt(torch.sum((points_expanded_i - points_expanded_j) ** 2, dim=-1))
    
    return distance_matrices


def calculate_tour_cost_batch_pomo(tours, distance_matrices):
    """
    POMO版本的批量路径成本计算
    Args:
        tours: torch.Tensor of shape (batch_size * num_nodes, num_nodes + 1) - 路径张量
        distance_matrices: torch.Tensor of shape (batch_size, num_nodes, num_nodes) - 距离矩阵
    Returns:
        costs: torch.Tensor of shape (batch_size * num_nodes,) - 每个路径的成本
    """
    batch_size, num_nodes, _ = distance_matrices.shape
    total_tours = tours.shape[0]  # batch_size * num_nodes
    device = tours.device
    
    # 扩展距离矩阵以匹配tours的维度
    # 从 (batch_size, num_nodes, num_nodes) 扩展为 (batch_size * num_nodes, num_nodes, num_nodes)
    expanded_distances = distance_matrices.unsqueeze(1).expand(-1, num_nodes, -1, -1)
    expanded_distances = expanded_distances.reshape(total_tours, num_nodes, num_nodes)
    
    # 计算路径成本
    costs = torch.zeros(total_tours, device=device)
    
    # 使用张量操作计算所有路径段的成本
    for i in range(tours.shape[1] - 1):  # num_nodes + 1 - 1 = num_nodes 个路径段
        current_nodes = tours[:, i]      # shape: (total_tours,)
        next_nodes = tours[:, i + 1]     # shape: (total_tours,)
        
        # 获取对应的距离
        batch_indices = torch.arange(total_tours, device=device)
        segment_costs = expanded_distances[batch_indices, current_nodes, next_nodes]
        costs += segment_costs
    
    return costs


def calculate_tour_cost_batch(tours, distance_matrices):
    """
    原始版本的批量计算路径成本（保持兼容性）
    Args:
        tours: list of lists - 每个样本的路径
        distance_matrices: torch.Tensor of shape (batch_size, num_nodes, num_nodes)
    Returns:
        costs: torch.Tensor of shape (batch_size,)
    """
    batch_size = len(tours)
    device = distance_matrices.device
    costs = []
    
    for b in range(batch_size):
        tour = tours[b]
        distance_matrix = distance_matrices[b]
        total_cost = 0.0
        
        for i in range(len(tour) - 1):
            total_cost += distance_matrix[tour[i], tour[i + 1]]
        
        costs.append(total_cost)
    
    return torch.stack(costs)


class TSPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

    self.train_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.training_split),
        sparse_factor=self.args.sparse_factor,
    )

    # 创建测试集并只保留前1000条数据
    self.test_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.test_split),
        sparse_factor=self.args.sparse_factor,
    )
    self.test_dataset.file_lines = self.test_dataset.file_lines[:1000]

    self.validation_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        sparse_factor=self.args.sparse_factor,
    )
    
    # 强化学习相关参数
    self.rl_loss_weight = getattr(self.args, 'rl_loss_weight', 0.1)  # 强化学习损失权重
    self.rl_baseline_decay = getattr(self.args, 'rl_baseline_decay', 0.95)  # 基线衰减率
    self.rl_baseline = None  # 用于减少方差的基线
    self.rl_compute_frequency = getattr(self.args, 'rl_compute_frequency', 1)  # 每隔多少步计算一次RL损失
    self.pomo_temperature = getattr(self.args, 'pomo_temperature', 1.0)  # POMO求解器的温度参数，控制探索程度

  def forward(self, x, adj, t, edge_index):
    return self.model(x, t, adj, edge_index)

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    _, points, adj_matrix, gt_tour = batch
    t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)

    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    if self.sparse:
      adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

    xt = self.diffusion.sample(adj_matrix_onehot, t)
    xt = xt * 2 - 1
    xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

    # Denoise
    x0_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index,
    )

    # Compute standard cross-entropy loss
    loss_func = nn.CrossEntropyLoss()
    ce_loss = loss_func(x0_pred, adj_matrix.long())
    
    # 强化学习辅助损失计算
    rl_loss = torch.tensor(0.0, device=points.device)
    
    # 只在指定频率下计算RL损失以节省计算资源
    compute_rl = (self.rl_loss_weight > 0 and 
                  not self.sparse and 
                  batch_idx % self.rl_compute_frequency == 0)
    
    if compute_rl:  # 暂时只支持非稀疏图
        # 获取预测的邻接矩阵概率
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        # 取边存在的概率 (第二个类别)
        adj_prob_matrix = x0_pred_prob[:, :, :, 1]  # shape: (batch_size, num_nodes, num_nodes)
        
        try:
            # 根据参数选择使用POMO还是单一起始点
            use_pomo = getattr(self.args, 'use_pomo', False)
            
            if use_pomo:
                # 使用POMO版本的求解器获取路径和对数概率，支持探索
                pred_tours, log_probs = greedy_tsp_solver_batch_pomo(adj_prob_matrix, temperature=self.pomo_temperature)
                # pred_tours: (batch_size * num_nodes, num_nodes + 1)
                # log_probs: (batch_size * num_nodes,)
                
                # 计算真实距离矩阵
                distance_matrices = calculate_euclidean_distance_batch(points)
                
                # 计算预测路径的成本
                pred_costs = calculate_tour_cost_batch_pomo(pred_tours, distance_matrices)
                # pred_costs: (batch_size * num_nodes,)
                
                # 计算真实最优路径的成本作为基准
                gt_tours_list = []
                for b in range(points.shape[0]):
                    gt_tour_b = gt_tour[b].cpu().numpy().tolist()
                    gt_tour_b.append(gt_tour_b[0])  # 添加回到起点
                    gt_tours_list.append(gt_tour_b)
                
                gt_costs = calculate_tour_cost_batch(gt_tours_list, distance_matrices)
                # gt_costs: (batch_size,)
                
                # 将gt_costs扩展以匹配POMO的维度
                num_nodes = points.shape[1] 
                gt_costs_expanded = gt_costs.unsqueeze(1).expand(-1, num_nodes).reshape(-1)
                # gt_costs_expanded: (batch_size * num_nodes,)
                
                # 计算奖励 (负的相对成本差异)
                relative_cost_diff = (pred_costs - gt_costs_expanded) / (gt_costs_expanded + 1e-8)
                rewards = -relative_cost_diff  # 成本越低，奖励越高
                
                # 对于POMO，我们选择每个样本中最好的路径来计算基线
                rewards_reshaped = rewards.reshape(points.shape[0], num_nodes)  # (batch_size, num_nodes)
                best_rewards = torch.max(rewards_reshaped, dim=1)[0]  # (batch_size,)
                
                # 更新基线 (使用指数移动平均)
                current_baseline = best_rewards.mean().detach()
                if self.rl_baseline is None:
                    self.rl_baseline = current_baseline
                else:
                    self.rl_baseline = self.rl_baseline_decay * self.rl_baseline + (1 - self.rl_baseline_decay) * current_baseline
                
                # 计算优势函数 (奖励减去基线)
                advantages = rewards - self.rl_baseline
                
                # REINFORCE损失 (负的对数概率乘以优势)
                rl_loss = -(log_probs * advantages.detach()).mean()
                
                # 记录强化学习相关指标
                self.log("train/rl_loss", rl_loss)
                self.log("train/avg_reward", rewards.mean())
                self.log("train/best_reward", best_rewards.mean())
                self.log("train/baseline", self.rl_baseline)
                self.log("train/avg_pred_cost", pred_costs.mean())
                self.log("train/best_pred_cost", pred_costs.reshape(points.shape[0], num_nodes).min(dim=1)[0].mean())
                self.log("train/avg_gt_cost", gt_costs.mean())
                self.log("train/cost_gap_percent", (relative_cost_diff * 100).mean())
                self.log("train/best_cost_gap_percent", ((pred_costs.reshape(points.shape[0], num_nodes).min(dim=1)[0] - gt_costs) / (gt_costs + 1e-8) * 100).mean())
                self.log("train/pomo_temperature", self.pomo_temperature)
                
            else:
                # 使用原始版本的贪婪求解器
                pred_tours, log_probs = greedy_tsp_solver_batch(adj_prob_matrix, start_node=0)
                
                # 计算真实距离矩阵
                distance_matrices = calculate_euclidean_distance_batch(points)
                
                # 计算预测路径的成本
                pred_costs = calculate_tour_cost_batch(pred_tours, distance_matrices)
                
                # 计算真实最优路径的成本作为基准
                gt_tours_list = []
                for b in range(points.shape[0]):
                    gt_tour_b = gt_tour[b].cpu().numpy().tolist()
                    gt_tour_b.append(gt_tour_b[0])  # 添加回到起点
                    gt_tours_list.append(gt_tour_b)
                
                gt_costs = calculate_tour_cost_batch(gt_tours_list, distance_matrices)
                
                # 计算奖励 (负的相对成本差异)
                relative_cost_diff = (pred_costs - gt_costs) / (gt_costs + 1e-8)
                rewards = -relative_cost_diff  # 成本越低，奖励越高
                
                # 更新基线 (使用指数移动平均)
                current_baseline = rewards.mean().detach()
                if self.rl_baseline is None:
                    self.rl_baseline = current_baseline
                else:
                    self.rl_baseline = self.rl_baseline_decay * self.rl_baseline + (1 - self.rl_baseline_decay) * current_baseline
                
                # 计算优势函数 (奖励减去基线)
                advantages = rewards - self.rl_baseline
                
                # REINFORCE损失 (负的对数概率乘以优势)
                rl_loss = -(log_probs * advantages.detach()).mean()
                
                # 记录强化学习相关指标
                self.log("train/rl_loss", rl_loss)
                self.log("train/avg_reward", rewards.mean())
                self.log("train/baseline", self.rl_baseline)
                self.log("train/avg_pred_cost", pred_costs.mean())
                self.log("train/avg_gt_cost", gt_costs.mean())
                self.log("train/cost_gap_percent", (relative_cost_diff * 100).mean())
                self.log("train/pomo_temperature", self.pomo_temperature)
            
        except Exception as e: 
            print(f"强化学习损失计算失败: {e}")
            rl_loss = torch.tensor(0.0, device=points.device)
    
    # 总损失 = 交叉熵损失 + 强化学习损失
    total_loss = ce_loss + self.rl_loss_weight * rl_loss
    
    # 记录核心损失和指标
    self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True)
    self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)
    self.log("train/rl_loss_weight", self.rl_loss_weight, on_step=False, on_epoch=True)
    
    # 记录训练过程中的重要信息
    self.log("train/batch_idx", float(batch_idx), on_step=True, on_epoch=False)
    self.log("train/batch_size", float(points.shape[0]), on_step=True, on_epoch=False)
    
    # 记录扩散时间步信息
    avg_t = np.mean(np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int))
    self.log("train/avg_diffusion_timestep", avg_t, on_step=True, on_epoch=True)
    
    return total_loss

  def gaussian_training_step(self, batch, batch_idx):
    if self.sparse:
      # TODO: Implement Gaussian diffusion with sparse graphs
      raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
    _, points, adj_matrix, _ = batch

    adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    # Sample from diffusion
    t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
    xt, epsilon = self.diffusion.sample(adj_matrix, t)

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    # Denoise
    epsilon_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        None,
    )
    epsilon_pred = epsilon_pred.squeeze(1)

    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss)
    return loss

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1).to(device)
      x0_pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )

      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      else:
        x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt

  def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1).to(device)
      pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def test_step(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour = batch
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
    else:
      real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      points = points.reshape((-1, 2))
      edge_index = edge_index.reshape((2, -1))
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      np_edge_index = edge_index.cpu().numpy()

    stacked_tours = []
    # ns, merge_iterations = 0, 0

    if self.args.parallel_sampling > 1:
      if not self.sparse:
        points = points.repeat(self.args.parallel_sampling, 1, 1)
      else:
        points = points.repeat(self.args.parallel_sampling, 1)
        edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        else:
          xt = xt.repeat(self.args.parallel_sampling, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        if self.diffusion_type == 'gaussian':
          xt = self.gaussian_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)
        else:
          xt = self.categorical_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)

      if self.diffusion_type == 'gaussian':
        adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        adj_mat = xt.float().cpu().detach().numpy() + 1e-6

      if self.args.save_numpy_heatmap:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

      # 使用强化学习方法根据扩散模型预测的热力图生成TSP解决方案
      try:
        if not self.sparse:  # 目前只支持非稀疏图
          # 将numpy数组转换为torch tensor
          adj_prob_matrix = torch.from_numpy(adj_mat).float().to(device)
          
          # 确保是单个样本的形状 (1, num_nodes, num_nodes)
          if adj_prob_matrix.dim() == 2:
            adj_prob_matrix = adj_prob_matrix.unsqueeze(0)
          
          # 使用POMO方法生成多个候选路径
          use_pomo = getattr(self.args, 'use_pomo', True)  # 测试时默认使用POMO
          test_temperature = getattr(self.args, 'test_temperature', 0.0)  # 测试时使用较小的温度以减少随机性
          
          if use_pomo:
            # 使用POMO版本生成路径
            pred_tours, _ = greedy_tsp_solver_batch_pomo(adj_prob_matrix, temperature=test_temperature)
            # pred_tours: (1 * num_nodes, num_nodes + 1)
            
            # 计算所有路径的成本
            points_tensor = torch.from_numpy(np_points).float().unsqueeze(0).to(device)  # (1, num_nodes, 2)
            distance_matrices = calculate_euclidean_distance_batch(points_tensor)
            pred_costs = calculate_tour_cost_batch_pomo(pred_tours, distance_matrices)
            
            # 选择成本最小的路径
            best_idx = torch.argmin(pred_costs)
            best_tour = pred_tours[best_idx].cpu().numpy()
            
            # 转换为列表格式，去掉最后的重复起始点
            solved_tours = [best_tour[:-1].tolist()]
            
          else:
            # 使用原始版本的贪婪求解器
            pred_tours, _ = greedy_tsp_solver_batch(adj_prob_matrix, start_node=0)
            
            # 转换为numpy数组格式
            solved_tours = []
            for tour in pred_tours:
              # 去掉最后的重复起始点
              solved_tours.append(tour[:-1])
          
          # 转换为numpy数组
          solved_tours = np.array(solved_tours, dtype='int64')
          
        else:
          # 对于稀疏图，暂时回退到原来的方法或抛出异常
          raise NotImplementedError("强化学习方法暂不支持稀疏图，请设置sparse=False")
          
      except Exception as e:
        print(f"强化学习TSP求解失败: {e}")
        print("回退到原始merge_tours方法")
        # 回退到原始方法
        tours, merge_iterations = merge_tours(
            adj_mat, np_points, np_edge_index,
            sparse_graph=self.sparse,
            parallel_sampling=self.args.parallel_sampling,
        )
        
        # Refine using 2-opt
        solved_tours, ns = batched_two_opt_torch(
            np_points.astype("float64"), np.array(tours).astype('int64'),
            max_iterations=self.args.two_opt_iterations, device=device)

      stacked_tours.append(solved_tours)

    solved_tours = np.concatenate(stacked_tours, axis=0)

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)

    total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
    all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)]
    best_solved_cost = np.min(all_solved_costs)

    # 可视化对比真实路径和预测路径
    debug_mode = True
    if debug_mode: 
        import matplotlib.pyplot as plt
        
        # 只可视化第一个样本以避免生成过多图片
        points = np_points  # shape: (num_nodes, 2) 
        gt_path = np_gt_tour # shape: (num_nodes,)
        pred_path = solved_tours[0]  # shape: (num_nodes,)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制真实路径
        ax1.scatter(points[:, 0], points[:, 1], c='blue', s=50)
        for i in range(len(gt_path)):
            start = points[gt_path[i]]
            end = points[gt_path[(i + 1) % len(gt_path)]]
            ax1.plot([start[0], end[0]], [start[1], end[1]], 'r-')
        ax1.set_title(f'真实路径 (成本: {gt_cost:.2f})')
        
        # 绘制预测路径
        ax2.scatter(points[:, 0], points[:, 1], c='blue', s=50)
        for i in range(len(pred_path)):
            start = points[pred_path[i]]
            end = points[pred_path[(i + 1) % len(pred_path)]]
            ax2.plot([start[0], end[0]], [start[1], end[1]], 'g-')
        ax2.set_title(f'预测路径 (成本: {best_solved_cost:.2f})')
        
        # 保存图片
        exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, str(self.logger.version))
        vis_path = os.path.join(exp_save_dir, 'route_visualization')
        os.makedirs(vis_path, exist_ok=True)
        plt.savefig(os.path.join(vis_path, f'route_comparison_{split}_{batch_idx}.png'))
        plt.close() 
    

    # 计算额外的性能指标
    gap_percentage = ((best_solved_cost - gt_cost) / gt_cost) * 100
    all_gaps = [((cost - gt_cost) / gt_cost) * 100 for cost in all_solved_costs]
    avg_gap = np.mean(all_gaps)
    std_gap = np.std(all_gaps)
    
    metrics = {
        f"{split}/gt_cost": gt_cost,
        f"{split}/gap_percentage": gap_percentage,
        f"{split}/avg_gap_percentage": avg_gap,
        f"{split}/std_gap_percentage": std_gap,
        # f"{split}/2opt_iterations": ns,  # 注释掉因为不再使用2-opt
        # f"{split}/merge_iterations": merge_iterations,  # 注释掉因为不再使用merge_tours
        f"{split}/total_sampling": total_sampling,
        f"{split}/sequential_sampling": self.args.sequential_sampling,
        f"{split}/parallel_sampling": self.args.parallel_sampling,
    }
    
    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    
    # 特别标记最重要的指标用于进度条显示，统一参数以避免重复记录错误
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
    
    return metrics

  def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split):
    if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
