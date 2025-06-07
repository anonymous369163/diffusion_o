"""
扩散模型 + 强化学习的TSP训练器
结合difusco扩散模型和POMO强化学习方法
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from logging import getLogger
import logging
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

# 添加TensorBoard相关导入
from torch.utils.tensorboard import SummaryWriter
import time
from collections import deque

# 导入扩散模型相关
sys.path.append('/root/code/difusco_o/difusco_o/DIFUSCO/difusco')
from difusco.utils.diffusion_schedulers import InferenceSchedule
# from difusco.pl_tsp_model import TSPModel
# from difusco.utils.draw_utils import visualize_tsp_solutions

# 导入强化学习相关
sys.path.append('/root/code/diffusco_o/difusco_o/DIFUSCO/mtnco/MTPOMO/POMO')
from mtnco.utils.utils import *


class TSPEnv:
    """TSP环境，基于扩散模型生成的邻接矩阵进行强化学习"""
    
    def __init__(self, batch_size, num_nodes, pomo_size=None):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.pomo_size = pomo_size if pomo_size else num_nodes
        
        # 状态变量
        self.points = None
        self.adj_matrix = None
        self.visited_mask = None
        self.current_node = None
        self.tour = None
        self.tour_length = None
        self.done = None
        
        # 批次和POMO索引
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        
    def reset(self, points, adj_matrix):
        """
        重置环境状态
        Args:
            points: (batch_size, num_nodes, 2) - 节点坐标
            adj_matrix: (batch_size, num_nodes, num_nodes) - 扩散模型生成的邻接矩阵
        """
        self.points = points
        self.adj_matrix = adj_matrix
        
        # 获取设备信息，确保所有张量在同一设备上
        device = points.device
        
        # 初始化状态（确保在正确的设备上）
        self.visited_mask = torch.zeros(self.batch_size, self.pomo_size, self.num_nodes, dtype=torch.bool, device=device)
        self.current_node = torch.zeros(self.batch_size, self.pomo_size, dtype=torch.long, device=device)
        self.tour = [torch.zeros(self.batch_size, self.pomo_size, dtype=torch.long, device=device)]
        self.tour_length = torch.zeros(self.batch_size, self.pomo_size, device=device)
        self.done = torch.zeros(self.batch_size, self.pomo_size, dtype=torch.bool, device=device)
        
        # 确保BATCH_IDX和POMO_IDX也在正确的设备上
        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(self.batch_size, self.pomo_size)
        
        # 标记起始节点为已访问 - 避免原地操作
        new_visited_mask = self.visited_mask.clone()
        new_visited_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = True
        self.visited_mask = new_visited_mask
        
        return self.get_state()
    
    def step(self, action):
        """
        执行动作
        Args:
            action: (batch_size, pomo_size) - 选择的下一个节点
        """
        # 更新当前节点
        prev_node = self.current_node.clone()
        self.current_node = action
        
        # 计算移动距离（基于真实欧几里得距离）
        step_distance = self.calculate_distance(prev_node, self.current_node)
        self.tour_length += step_distance
        
        # 标记节点为已访问 - 避免原地操作
        new_visited_mask = self.visited_mask.clone()
        new_visited_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = True
        self.visited_mask = new_visited_mask
        
        # 添加到路径
        self.tour.append(self.current_node.clone())
        
        # 检查是否完成
        if len(self.tour) == self.num_nodes:
            # 回到起始节点（确保在正确的设备上）
            final_distance = self.calculate_distance(self.current_node, torch.zeros_like(self.current_node))
            self.tour_length += final_distance
            self.done = torch.ones_like(self.done)
            
            # 计算奖励（负的路径长度）
            reward = -self.tour_length
        else:
            # 确保reward在正确的设备上
            reward = torch.zeros(self.batch_size, self.pomo_size, device=self.points.device)
        
        return self.get_state(), reward, self.done
    
    def get_state(self):
        """获取当前状态"""
        return {
            'current_node': self.current_node,
            'visited_mask': self.visited_mask,
            'adj_matrix': self.adj_matrix,
            'points': self.points,
            'tour_length': self.tour_length,
            'done': self.done
        }
    
    def calculate_distance(self, node1, node2):
        """计算两个节点之间的欧几里得距离"""
        # node1, node2: (batch_size, pomo_size)
        batch_size, pomo_size = node1.shape
        
        # 获取节点坐标
        node1_coords = self.points[self.BATCH_IDX, node1]  # (batch_size, pomo_size, 2)
        node2_coords = self.points[self.BATCH_IDX, node2]  # (batch_size, pomo_size, 2)
        
        # 计算欧几里得距离
        distance = torch.sqrt(((node1_coords - node2_coords) ** 2).sum(dim=-1))
        return distance
    
    def get_available_actions_mask(self):
        """获取可用动作的掩码"""
        # 返回未访问节点的掩码
        return ~self.visited_mask


class TSPRLModel(nn.Module):
    """基于邻接矩阵的TSP强化学习模型（可选简单神经网络增强）"""
    
    def __init__(self, 
                 use_neural_network=True,
                 network_type='linear',  # 'linear', 'mlp', 'none'
                 embedding_dim=64,
                 hidden_dim=128):
        super().__init__()
        
        self.use_neural_network = use_neural_network
        self.network_type = network_type
        self.embedding_dim = embedding_dim
        
        if self.use_neural_network and self.network_type != 'none':
            if self.network_type == 'linear':
                # 简单线性层：节点坐标 -> 嵌入
                self.node_embedding = nn.Linear(2, embedding_dim)
                self.context_projection = nn.Linear(embedding_dim, 1)
                
            elif self.network_type == 'mlp':
                # 简单MLP：节点坐标 -> 隐藏层 -> 嵌入
                self.node_embedding = nn.Sequential(
                    nn.Linear(2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                )
                self.context_projection = nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
        
        print(f"TSPRLModel初始化: use_neural_network={use_neural_network}, network_type={network_type}")
        
    def forward(self, state):
        """
        前向传播 - 以邻接矩阵为主要决策依据
        Args:
            state: 环境状态字典
        Returns:
            logits: (batch_size, pomo_size, num_nodes) - 动作概率logits
        """
        points = state['points']
        current_node = state['current_node']
        visited_mask = state['visited_mask']
        adj_matrix = state['adj_matrix']
        
        batch_size, pomo_size = current_node.shape
        num_nodes = points.shape[1]
        
        # 动态创建批次和POMO索引（用于高级索引）
        BATCH_IDX = torch.arange(batch_size, device=current_node.device)[:, None].expand(batch_size, pomo_size)
        # POMO_IDX = torch.arange(pomo_size, device=current_node.device)[None, :].expand(batch_size, pomo_size)
        
        # 主要决策依据：从邻接矩阵获取权重
        # adj_matrix: (batch_size, num_nodes, num_nodes)
        # 获取从当前节点到所有节点的连接权重
        adj_logits = adj_matrix[BATCH_IDX, current_node]  # (batch_size, pomo_size, num_nodes)
        
        # 可选的神经网络增强
        if self.use_neural_network and self.network_type != 'none':
            # 获取节点特征
            node_features = self.node_embedding(points)  # (batch_size, num_nodes, embedding_dim)
            
            # 获取当前节点特征
            current_features = node_features[BATCH_IDX, current_node]  # (batch_size, pomo_size, embedding_dim)
            
            # 计算上下文权重
            # context_weights = self.context_projection(current_features)  # (batch_size, pomo_size, 1)
            
            # 计算所有节点的特征匹配度
            # 使用简单的点积计算相似度
            node_features_expanded = node_features.unsqueeze(1).expand(batch_size, pomo_size, num_nodes, self.embedding_dim)
            current_features_expanded = current_features.unsqueeze(2).expand(batch_size, pomo_size, num_nodes, self.embedding_dim)
            
            # 计算特征相似度
            feature_similarity = (node_features_expanded * current_features_expanded).sum(dim=-1)  # (batch_size, pomo_size, num_nodes)
            
            # 结合邻接矩阵和特征相似度
            # 邻接矩阵权重更大（主导作用）
            alpha = 0.8  # 邻接矩阵权重
            beta = 0.2   # 特征相似度权重
            
            logits = alpha * adj_logits + beta * feature_similarity
            
        else:
            # 只使用邻接矩阵
            logits = adj_logits
        
        # 应用访问掩码（已访问的节点设置为负无穷）- 使用torch.where避免梯度问题
        masked_logits = torch.where(visited_mask, 
                                  torch.full_like(logits, float('-inf')), 
                                  logits)
        
        return masked_logits


class DiffusionRLTrainer:
    """扩散模型+强化学习训练器"""
    
    def __init__(self, 
                 diffusion_model_path,
                 num_nodes=50,
                 batch_size=32,
                 pomo_size=50,
                 lr=1e-4,
                 device='cuda',
                 # 新增超参数
                 use_neural_network=True,
                 network_type='linear',  # 'linear', 'mlp', 'none'
                 embedding_dim=64,
                 timestamp=None):
        
        self.device = device
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.pomo_size = pomo_size
        
        # 初始化扩散模型
        self.diffusion_model = self.load_diffusion_model(diffusion_model_path)
        
        # 初始化强化学习模型（以邻接矩阵为主）
        self.rl_model = TSPRLModel(
            use_neural_network=use_neural_network,
            network_type=network_type,
            embedding_dim=embedding_dim
        ).to(device)
        
        # 初始化环境
        self.env = TSPEnv(batch_size, num_nodes, pomo_size)
        
        # 初始化优化器
        if use_neural_network and network_type != 'none':
            # 如果使用神经网络，同时优化扩散模型和RL模型
            all_params = list(self.diffusion_model.parameters()) + list(self.rl_model.parameters())
        else:
            # 如果不使用神经网络，只优化扩散模型
            all_params = list(self.diffusion_model.parameters())
            
        self.optimizer = Optimizer(all_params, lr=lr)
        self.scheduler = Scheduler(self.optimizer, milestones=[100, 200], gamma=0.1)
        
        # 日志
        self.logger = getLogger(name='diffusion_rl_trainer')
        
        # TensorBoard设置
        if timestamp is None:   
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = f'runs/diffusion_rl_tsp_{timestamp}'
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger.info(f"TensorBoard日志保存到: {log_dir}")
        self.logger.info("启动TensorBoard: tensorboard --logdir=runs")
        
        # 模型保存目录设置
        self.save_dir = f'saved_models/diffusion_rl_tsp_{timestamp}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"模型保存目录: {self.save_dir}")
        
        # 训练统计
        self.global_step = 0
        self.best_tour_length = float('inf')
        self.best_eval_length = float('inf')  # 跟踪最佳评估结果
        self.recent_losses = deque(maxlen=100)  # 保存最近100个损失值
        self.recent_tour_lengths = deque(maxlen=100)  # 保存最近100个路径长度
        
    def load_diffusion_model(self, model_path):
        """加载扩散模型"""
        from difusco.difusion_tool import arg_parser, TSPModel_v2
        
        args = arg_parser()
        args.ckpt_path = model_path
        args.resume_weight_only = True
        
        model = TSPModel_v2.load_from_checkpoint(args.ckpt_path, param_args=args)
        model = model.to(self.device)
        model.eval()
        
        # 添加tensor版本的生成方法
        def generate_adj_tensor(points):
            """
            生成邻接矩阵的Tensor版本，保持梯度连接
            Args:
                points: (batch_size, num_nodes, 2) - 节点坐标
            Returns:
                adj_matrix: (batch_size, num_nodes, num_nodes) - 邻接矩阵
            """
            # 确保points在正确设备上
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).float().to(self.device)
            
            batch_size, num_nodes, _ = points.shape
            
            # 初始化随机噪声矩阵（保持梯度）
            xt = torch.randn(batch_size, num_nodes, num_nodes, device=self.device, requires_grad=True)
            
            if model.diffusion_type == 'gaussian':
                xt.requires_grad_(True)
            else:
                xt = (xt > 0).long()
            
            steps = model.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=model.args.inference_schedule,
                T=model.diffusion.T, 
                inference_T=steps
            )
            
            # 创建不使用no_grad的去噪步骤
            def gaussian_denoise_step_with_grad(points, xt, t, device, edge_index=None, target_t=None):
                """保持梯度的高斯去噪步骤"""
                t_tensor = torch.from_numpy(t).view(1).to(device)
                epsilon_pred = model.forward(
                    points.float().to(device),
                    xt.float().to(device),
                    t_tensor.float().to(device),
                    edge_index.long().to(device) if edge_index is not None else None,
                )
                epsilon_pred = epsilon_pred.squeeze(1)
                xt = model.gaussian_posterior(target_t, t_tensor, epsilon_pred, xt)
                return xt
            
            def categorical_denoise_step_with_grad(points, xt, t, device, edge_index=None, target_t=None):
                """保持梯度的分类去噪步骤"""
                t_tensor = torch.from_numpy(t).view(1).to(device)
                x0_pred = model.forward(
                    points.float().to(device),
                    xt.float().to(device),
                    t_tensor.float().to(device),
                    edge_index.long().to(device) if edge_index is not None else None,
                )
                
                if not model.sparse:
                    x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                else:
                    x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)
                
                xt = model.categorical_posterior(target_t, t_tensor, x0_pred_prob, xt)
                return xt
            
            # 扩散迭代（不使用torch.no_grad()）
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)
                
                if model.diffusion_type == 'gaussian':
                    xt = gaussian_denoise_step_with_grad(points, xt, t1, self.device, None, target_t=t2)
                else:
                    xt = categorical_denoise_step_with_grad(points, xt, t1, self.device, None, target_t=t2)
            
            # 最终处理（保持梯度）
            if model.diffusion_type == 'gaussian':
                adj_matrix = xt * 0.5 + 0.5
            else:
                adj_matrix = xt.float() + 1e-6
            
            return adj_matrix
        
        # 将方法绑定到模型
        model.generate_adj_tensor = generate_adj_tensor
        
        return model
    
    def generate_problems(self, batch_size):
        """生成TSP问题实例"""
        # 生成随机点
        points = torch.rand(batch_size, self.num_nodes, 2, device=self.device)
        return points
    
    def train_one_batch(self):
        """训练一个批次"""
        # 生成问题实例
        points = self.generate_problems(self.batch_size)
        
        # 使用扩散模型生成邻接矩阵 - 保持梯度连接
        # 移除torch.no_grad()以确保梯度能传播到扩散模型参数
        adj_matrix = self.diffusion_model.generate_adj_tensor(points)  # 使用tensor版本
        
        # 验证梯度连接
        # print(f"adj_matrix.requires_grad: {adj_matrix.requires_grad}")
        
        # 重置环境
        state = self.env.reset(points, adj_matrix)
        
        # 收集轨迹
        log_probs = []
        rewards = []
        
        # POMO rollout
        for step in range(self.num_nodes - 1):
            # 获取动作概率（主要基于邻接矩阵）
            # 这里adj_matrix通过rl_model参与计算，建立梯度连接
            logits = self.rl_model(state)
            
            # 验证logits的梯度连接
            # if step == 0:  # 只在第一步打印，避免过多输出
            #     print(f"logits.requires_grad: {logits.requires_grad}")
            
            probs = F.softmax(logits, dim=-1)
            
            # 采样动作
            action_dist = torch.distributions.Categorical(probs)
            actions = action_dist.sample()
            
            # 记录log概率
            log_prob = action_dist.log_prob(actions)
            log_probs.append(log_prob)
            
            # 执行动作
            state, reward, done = self.env.step(actions)
            
            if done.all():
                rewards.append(reward)
                break
        
        # 计算总奖励
        total_rewards = rewards[-1] if rewards else torch.zeros(self.batch_size, self.pomo_size, device=self.device)
        
        # 获取最佳路径信息用于可视化
        best_tour_info = self.get_best_tour_info(points, adj_matrix, total_rewards)
        
        # 计算advantage
        baseline = total_rewards.mean(dim=1, keepdim=True)
        advantage = total_rewards - baseline
        
        # 计算总log概率
        total_log_prob = torch.stack(log_probs).sum(dim=0)
        
        # 计算REINFORCE损失
        # 现在这个损失包含了从扩散模型参数到adj_matrix到reward的完整计算图
        rl_loss = -(advantage * total_log_prob).mean()
        
        # 验证损失的梯度连接
        # print(f"rl_loss.requires_grad: {rl_loss.requires_grad}")
        
        # 反向传播 - 现在梯度可以正确传播到扩散模型参数
        self.optimizer.zero_grad()
        rl_loss.backward()
        
        # 记录梯度信息
        diffusion_grad_norm = 0.0
        rl_grad_norm = 0.0
        
        # 计算扩散模型梯度范数
        for p in self.diffusion_model.parameters():
            if p.grad is not None:
                diffusion_grad_norm += p.grad.data.norm(2).item() ** 2
        diffusion_grad_norm = diffusion_grad_norm ** 0.5
        
        # 计算RL模型梯度范数
        for p in self.rl_model.parameters():
            if p.grad is not None:
                rl_grad_norm += p.grad.data.norm(2).item() ** 2
        rl_grad_norm = rl_grad_norm ** 0.5
        
        # 验证扩散模型参数是否有梯度
        # has_diffusion_grad = any(p.grad is not None for p in self.diffusion_model.parameters())
        # print(f"扩散模型参数有梯度: {has_diffusion_grad}")
        
        self.optimizer.step()
        
        # 计算统计信息
        best_rewards = total_rewards.max(dim=1)[0]
        avg_reward = best_rewards.mean().item()
        avg_tour_length = -avg_reward
        
        # 记录到TensorBoard
        self.log_to_tensorboard(avg_tour_length, rl_loss.item(), diffusion_grad_norm, rl_grad_norm, 
                               total_rewards, adj_matrix, points, best_tour_info)
        
        # 更新统计
        self.recent_losses.append(rl_loss.item())
        self.recent_tour_lengths.append(avg_tour_length)
        self.global_step += 1
        
        return avg_tour_length, rl_loss.item()
    
    def get_best_tour_info(self, points, adj_matrix, total_rewards):
        """获取最佳路径信息用于可视化"""
        with torch.no_grad():
            # 找到第一个batch中最佳的POMO实例
            batch_idx = 0
            best_pomo_idx = total_rewards[batch_idx].argmax().item()
            
            # 重新运行推理获取最佳路径（贪心解码）
            self.rl_model.eval()
            
            # 创建单个实例的环境来获取完整路径
            single_points = points[batch_idx:batch_idx+1]  # (1, num_nodes, 2)
            single_adj = adj_matrix[batch_idx:batch_idx+1]  # (1, num_nodes, num_nodes)
            
            # 临时创建单POMO环境
            temp_env = TSPEnv(1, self.num_nodes, 1)
            state = temp_env.reset(single_points, single_adj)
            
            tour_nodes = [0]  # 从节点0开始
            
            # 贪心解码获取完整路径
            for step in range(self.num_nodes - 1):
                logits = self.rl_model(state)
                action = logits.argmax(dim=-1)
                tour_nodes.append(action.item())
                state, _, done = temp_env.step(action)
                
                if done.all():
                    break
            
            # 回到起始点完成环路
            tour_nodes.append(0)
            
            self.rl_model.train()
            
            return {
                'tour_nodes': tour_nodes,
                'points': single_points[0].cpu().numpy(),
                'best_reward': total_rewards[batch_idx, best_pomo_idx].item()
            }
    
    def log_to_tensorboard(self, avg_tour_length, loss, diff_grad_norm, rl_grad_norm, 
                          total_rewards, adj_matrix, points, best_tour_info):
        """记录训练数据到TensorBoard"""
        
        # 基本训练指标
        self.writer.add_scalar('Training/Average_Tour_Length', avg_tour_length, self.global_step)
        self.writer.add_scalar('Training/Loss', loss, self.global_step)
        self.writer.add_scalar('Training/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # 梯度信息
        self.writer.add_scalar('Gradients/Diffusion_Model_Grad_Norm', diff_grad_norm, self.global_step)
        self.writer.add_scalar('Gradients/RL_Model_Grad_Norm', rl_grad_norm, self.global_step)
        
        # 奖励分布统计
        best_rewards = total_rewards.max(dim=1)[0]
        worst_rewards = total_rewards.min(dim=1)[0]
        mean_rewards = total_rewards.mean(dim=1)
        
        self.writer.add_scalar('Rewards/Best_Reward', best_rewards.mean().item(), self.global_step)
        self.writer.add_scalar('Rewards/Worst_Reward', worst_rewards.mean().item(), self.global_step)
        self.writer.add_scalar('Rewards/Mean_Reward', mean_rewards.mean().item(), self.global_step)
        self.writer.add_scalar('Rewards/Reward_Std', total_rewards.std().item(), self.global_step)
        
        # 更新最佳记录
        if avg_tour_length < self.best_tour_length:
            self.best_tour_length = avg_tour_length
            self.writer.add_scalar('Training/Best_Tour_Length', self.best_tour_length, self.global_step)
        
        # 邻接矩阵统计
        adj_mean = adj_matrix.mean().item()
        adj_std = adj_matrix.std().item()
        adj_max = adj_matrix.max().item()
        adj_min = adj_matrix.min().item()
        
        self.writer.add_scalar('Adjacency_Matrix/Mean', adj_mean, self.global_step)
        self.writer.add_scalar('Adjacency_Matrix/Std', adj_std, self.global_step)
        self.writer.add_scalar('Adjacency_Matrix/Max', adj_max, self.global_step)
        self.writer.add_scalar('Adjacency_Matrix/Min', adj_min, self.global_step)
        
        # 移动平均
        if len(self.recent_losses) > 10:
            self.writer.add_scalar('Training/Loss_MA', np.mean(list(self.recent_losses)), self.global_step)
            self.writer.add_scalar('Training/Tour_Length_MA', np.mean(list(self.recent_tour_lengths)), self.global_step)
        
        # 每100步可视化一个TSP实例
        if self.global_step % 100 == 0:
            self.visualize_tsp_solution(points[0], adj_matrix[0], total_rewards[0], best_tour_info)
    
    def visualize_tsp_solution(self, points, adj_matrix, rewards, best_tour_info):
        """可视化TSP解决方案"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 左图：节点分布和最佳求解路径
            points_np = points.cpu().numpy()
            tour_nodes = best_tour_info['tour_nodes']
            best_reward = best_tour_info['best_reward']
            
            # 绘制节点
            ax1.scatter(points_np[:, 0], points_np[:, 1], c='red', s=80, zorder=3)
            
            # 标注节点编号
            for i, (x, y) in enumerate(points_np):
                ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, fontweight='bold')
            
            # 绘制TSP路径
            tour_coords = points_np[tour_nodes]
            
            # 绘制路径线段
            for i in range(len(tour_nodes) - 1):
                start_point = tour_coords[i]
                end_point = tour_coords[i + 1]
                ax1.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                        'b-', linewidth=2, alpha=0.7, zorder=2)
                
                # 添加箭头表示方向
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                ax1.arrow(start_point[0] + 0.7*dx, start_point[1] + 0.7*dy, 
                         0.1*dx, 0.1*dy, head_width=0.02, head_length=0.02, 
                         fc='blue', ec='blue', zorder=2)
            
            # 高亮起始节点
            start_point = points_np[0]
            circle = plt.Circle((start_point[0], start_point[1]), 0.03, 
                              color='green', fill=True, zorder=4)
            ax1.add_patch(circle)
            ax1.annotate('START', (start_point[0], start_point[1]), 
                        xytext=(10, -15), textcoords='offset points',
                        fontsize=8, fontweight='bold', color='green')
            
            # 计算路径长度
            path_length = -best_reward
            
            ax1.set_title(f'TSP节点分布与求解路径\n路径长度: {path_length:.3f}')
            ax1.set_xlabel('X坐标')
            ax1.set_ylabel('Y坐标')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # 添加路径顺序信息
            path_str = ' → '.join([str(node) for node in tour_nodes[:6]]) + '...'
            ax1.text(0.02, 0.98, f'路径: {path_str}', transform=ax1.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 右图：邻接矩阵热力图
            adj_np = adj_matrix.cpu().detach().numpy()
            im = ax2.imshow(adj_np, cmap='viridis', interpolation='nearest')
            ax2.set_title('扩散模型生成的邻接矩阵')
            ax2.set_xlabel('目标节点')
            ax2.set_ylabel('源节点')
            
            # 在邻接矩阵上标出实际使用的路径
            for i in range(len(tour_nodes) - 1):
                from_node = tour_nodes[i]
                to_node = tour_nodes[i + 1]
                # 在邻接矩阵上画红色方框标记实际路径
                rect = patches.Rectangle((to_node-0.4, from_node-0.4), 0.8, 0.8, 
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax2.add_patch(rect)
            
            plt.colorbar(im, ax=ax2)
            
            # 添加总体信息
            best_reward_overall = rewards.max().item()
            fig.suptitle(f'步骤 {self.global_step} | 最佳奖励: {best_reward_overall:.3f} | 当前展示路径长度: {path_length:.3f}', 
                        fontsize=12)
            
            # plt.tight_layout()
            
            # 保存到TensorBoard
            self.writer.add_figure('Visualization/TSP_Instance_with_Solution', fig, self.global_step)
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"可视化失败: {e}")
            import traceback
            self.logger.warning(f"错误详情: {traceback.format_exc()}")
    
    def train(self, num_epochs=1000, log_interval=10):
        """训练主循环"""
        self.logger.info("开始训练扩散模型+强化学习...")
        self.logger.info(f"RL模型配置: use_neural_network={self.rl_model.use_neural_network}, network_type={self.rl_model.network_type}")
        self.logger.info(f"TensorBoard日志: {self.writer.log_dir}")
        
        # 记录超参数
        self.writer.add_hparams({
            'num_nodes': self.num_nodes,
            'batch_size': self.batch_size,
            'pomo_size': self.pomo_size,
            'lr': self.optimizer.param_groups[0]['lr'],
            'use_neural_network': self.rl_model.use_neural_network,
            'network_type': self.rl_model.network_type,
        }, {})
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练模式
            self.rl_model.train()
            self.diffusion_model.train()
            
            # 训练一个批次
            avg_length, total_loss = self.train_one_batch()
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录每个epoch的时间
            epoch_time = time.time() - epoch_start_time
            self.writer.add_scalar('Training/Epoch_Time', epoch_time, epoch)
            
            # 记录日志
            if epoch % log_interval == 0:
                elapsed_time = time.time() - start_time
                self.logger.info(f"Epoch {epoch:4d}: "
                               f"平均路径长度={avg_length:.4f}, "
                               f"总损失={total_loss:.4f}, "
                               f"最佳长度={self.best_tour_length:.4f}, "
                               f"用时={elapsed_time:.1f}s")
                
                # 记录进度信息
                self.writer.add_scalar('Training/Progress', epoch / num_epochs, epoch)
                self.writer.add_scalar('Training/Elapsed_Time', elapsed_time, epoch)
            
            # 定期评估
            if epoch % 50 == 0 and epoch > 0:
                self.logger.info(f"进行第{epoch}轮评估...")
                eval_length, is_best = self.evaluate(num_test_instances=100, visualize_solutions=True, max_visualizations=3)
                self.writer.add_scalar('Evaluation/Average_Tour_Length', eval_length, epoch)
                self.logger.info(f"评估完成，平均路径长度: {eval_length:.4f}")
                
                # 如果是最佳结果，保存模型
                if is_best:
                    self.save_best_model(eval_length, epoch)
            
            # 保存模型
            if epoch % 100 == 0 and epoch > 0:
                self.save_checkpoint(epoch)
        
        # 训练结束
        total_time = time.time() - start_time
        self.logger.info(f"训练完成！总用时: {total_time:.1f}s")
        self.logger.info(f"最佳路径长度: {self.best_tour_length:.4f}")
        
        # 最终评估
        final_eval_length, is_best_final = self.evaluate(num_test_instances=100, visualize_solutions=True, max_visualizations=5)
        self.writer.add_scalar('Final/Evaluation_Length', final_eval_length, num_epochs)
        
        # 如果最终评估也是最佳结果，保存模型
        if is_best_final:
            self.save_best_model(final_eval_length, num_epochs)
            self.logger.info("🎉 最终评估产生了历史最佳结果!")
        
        # 列出所有保存的模型文件
        self.logger.info("=" * 50)
        self.list_saved_models()
        self.logger.info("=" * 50)
        
        # 关闭TensorBoard writer
        self.writer.close()
    
    def save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'rl_model_state_dict': self.rl_model.state_dict(),
            'diffusion_model_state_dict': self.diffusion_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"保存检查点: {checkpoint_path}")
    
    def save_best_model(self, eval_length, epoch):
        """保存最佳模型参数"""
        best_model = {
            'epoch': epoch,
            'eval_length': eval_length,
            'best_eval_length': self.best_eval_length,
            'rl_model_state_dict': self.rl_model.state_dict(),
            'diffusion_model_state_dict': self.diffusion_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
        }
        
        best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        torch.save(best_model, best_model_path)
        self.logger.info(f"🎯 保存最佳模型: {best_model_path} (评估长度: {eval_length:.4f})")
        
        # 同时记录到TensorBoard
        self.writer.add_scalar('Best_Model/Eval_Length', eval_length, epoch)
        self.writer.add_scalar('Best_Model/Save_Epoch', epoch, epoch)
    
    def load_best_model(self, model_path=None):
        """加载最佳模型参数"""
        if model_path is None:
            model_path = os.path.join(self.save_dir, 'best_model.pt')
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.rl_model.load_state_dict(checkpoint['rl_model_state_dict'])
            self.diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.best_eval_length = checkpoint['best_eval_length']
            self.global_step = checkpoint['global_step']
            
            self.logger.info(f"✅ 成功加载最佳模型: {model_path}")
            self.logger.info(f"📊 模型评估长度: {checkpoint['eval_length']:.4f}")
            self.logger.info(f"🎯 训练轮次: {checkpoint['epoch']}")
            
            return True
            
        except FileNotFoundError:
            self.logger.warning(f"❌ 未找到最佳模型文件: {model_path}")
            return False
        except Exception as e:
            self.logger.error(f"❌ 加载最佳模型失败: {e}")
            return False
    
    def evaluate(self, num_test_instances=100, visualize_solutions=True, max_visualizations=5):
        """评估模型性能"""
        self.rl_model.eval()
        self.diffusion_model.eval()
        
        total_lengths = []
        eval_start_time = time.time()
        
        # 存储解决方案用于可视化
        solutions_to_visualize = []
        
        with torch.no_grad():
            for batch_idx in range(num_test_instances // self.batch_size):
                # 生成测试问题
                points = self.generate_problems(self.batch_size)
                
                # 生成邻接矩阵（评估时使用原始方法，不需要梯度）
                adj_matrix_np = self.diffusion_model.generate_adj(points.cpu().numpy())
                adj_matrix = torch.from_numpy(adj_matrix_np).float().to(self.device)
                
                # 重置环境
                state = self.env.reset(points, adj_matrix)
                
                # 记录初始状态用于路径追踪
                batch_tours = [[] for _ in range(self.batch_size)]
                for b in range(self.batch_size):
                    batch_tours[b].append(0)  # 起始节点
                
                # 贪心解码
                for step in range(self.num_nodes - 1):
                    logits = self.rl_model(state)
                    actions = logits.argmax(dim=-1)
                    
                    # 记录每个batch的路径
                    for b in range(self.batch_size):
                        # 取第一个POMO实例的行动（贪心最佳）
                        batch_tours[b].append(actions[b, 0].item())
                    
                    state, reward, done = self.env.step(actions)
                    
                    if done.all():
                        break
                
                # 完成环路
                for b in range(self.batch_size):
                    batch_tours[b].append(0)  # 回到起始节点
                
                # 收集最优路径长度
                best_lengths = state['tour_length'].max(dim=1)[0]
                total_lengths.extend(best_lengths.cpu().numpy())
                
                # 保存一些解决方案用于可视化
                if visualize_solutions and len(solutions_to_visualize) < max_visualizations:
                    for b in range(min(self.batch_size, max_visualizations - len(solutions_to_visualize))):
                        # 计算这个batch中最佳POMO实例的奖励
                        best_pomo_idx = state['tour_length'][b].argmax().item()
                        best_reward = -state['tour_length'][b, best_pomo_idx].item()
                        
                        solution_info = {
                            'points': points[b].cpu().numpy(),
                            'adj_matrix': adj_matrix[b].cpu().numpy(),
                            'tour_nodes': batch_tours[b],
                            'tour_length': -best_reward,
                            'batch_idx': batch_idx,
                            'instance_idx': b,
                            'rewards': state['tour_length'][b].cpu().numpy()  # 所有POMO实例的长度
                        }
                        solutions_to_visualize.append(solution_info)
        
        # 计算统计信息
        avg_length = np.mean(total_lengths)
        std_length = np.std(total_lengths)
        min_length = np.min(total_lengths)
        max_length = np.max(total_lengths)
        
        eval_time = time.time() - eval_start_time
        
        # 检查是否为最佳结果
        is_best = avg_length < self.best_eval_length
        if is_best:
            self.best_eval_length = avg_length
            self.logger.info(f"🏆 发现新的最佳评估结果! 平均路径长度: {avg_length:.4f}")
        
        # 可视化解决方案
        if visualize_solutions and solutions_to_visualize:
            self.logger.info(f"🎨 正在可视化 {len(solutions_to_visualize)} 个评估解决方案...")
            self.visualize_evaluation_solutions(solutions_to_visualize, avg_length, is_best)
        
        # 记录到TensorBoard
        if hasattr(self, 'writer'):
            self.writer.add_scalar('Evaluation/Average_Length', avg_length, self.global_step)
            self.writer.add_scalar('Evaluation/Std_Length', std_length, self.global_step)
            self.writer.add_scalar('Evaluation/Min_Length', min_length, self.global_step)
            self.writer.add_scalar('Evaluation/Max_Length', max_length, self.global_step)
            self.writer.add_scalar('Evaluation/Time', eval_time, self.global_step)
            self.writer.add_scalar('Evaluation/Best_Ever_Length', self.best_eval_length, self.global_step)
            
            # 添加长度分布直方图
            self.writer.add_histogram('Evaluation/Length_Distribution', np.array(total_lengths), self.global_step)
        
        self.logger.info(f"评估结果 - 平均路径长度: {avg_length:.4f} ± {std_length:.4f}")
        self.logger.info(f"评估结果 - 范围: [{min_length:.4f}, {max_length:.4f}]")
        self.logger.info(f"评估结果 - 历史最佳: {self.best_eval_length:.4f}")
        
        return avg_length, is_best
    
    def visualize_evaluation_solutions(self, solutions, avg_length, is_best):
        """可视化评估过程中的解决方案"""
        try:
            import matplotlib.pyplot as plt
            
            # 找到最佳和最差的解决方案
            best_solution = min(solutions, key=lambda x: x['tour_length'])
            worst_solution = max(solutions, key=lambda x: x['tour_length'])
            
            # 创建多子图展示
            n_solutions = min(len(solutions), 4)  # 最多显示4个解决方案
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            solutions_to_show = [best_solution, worst_solution] + solutions[:2] if len(solutions) > 2 else solutions
            
            for i, solution in enumerate(solutions_to_show[:n_solutions]):
                ax = axes[i]
                
                # 复用现有的可视化逻辑
                points_np = solution['points']
                tour_nodes = solution['tour_nodes']
                adj_matrix_np = solution['adj_matrix']
                tour_length = solution['tour_length']
                
                # 绘制节点
                ax.scatter(points_np[:, 0], points_np[:, 1], c='red', s=60, zorder=3)
                
                # 标注节点编号
                for j, (x, y) in enumerate(points_np):
                    ax.annotate(str(j), (x, y), xytext=(3, 3), textcoords='offset points', 
                              fontsize=7, fontweight='bold')
                
                # 绘制TSP路径
                tour_coords = points_np[tour_nodes]
                
                # 绘制路径线段
                for j in range(len(tour_nodes) - 1):
                    start_point = tour_coords[j]
                    end_point = tour_coords[j + 1]
                    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                           'b-', linewidth=1.5, alpha=0.7, zorder=2)
                
                # 高亮起始节点
                start_point = points_np[0]
                circle = plt.Circle((start_point[0], start_point[1]), 0.025, 
                                  color='green', fill=True, zorder=4)
                ax.add_patch(circle)
                
                # 设置标题和标签
                if i == 0 and solution == best_solution:
                    title_prefix = "🏆 最佳解"
                elif i == 1 and solution == worst_solution:
                    title_prefix = "📉 最差解"
                else:
                    title_prefix = f"解决方案 #{i+1}"
                    
                ax.set_title(f'{title_prefix}\n长度: {tour_length:.3f}', fontsize=10)
                ax.set_xlabel('X坐标', fontsize=8)
                ax.set_ylabel('Y坐标', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            
            # 隐藏未使用的子图
            for i in range(n_solutions, 4):
                axes[i].set_visible(False)
            
            # 设置总标题
            best_indicator = " 🏆 新最佳!" if is_best else ""
            fig.suptitle(f'评估解决方案可视化{best_indicator}\n平均长度: {avg_length:.4f} | 步骤: {self.global_step}', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存到TensorBoard
            if hasattr(self, 'writer'):
                self.writer.add_figure('Evaluation/Solutions_Visualization', fig, self.global_step)
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"评估可视化失败: {e}")
            import traceback
            self.logger.warning(f"错误详情: {traceback.format_exc()}")
    
    def list_saved_models(self):
        """列出保存目录中的所有模型文件"""
        if not os.path.exists(self.save_dir):
            self.logger.info(f"📁 保存目录不存在: {self.save_dir}")
            return []
        
        model_files = []
        for file in os.listdir(self.save_dir):
            if file.endswith('.pt'):
                file_path = os.path.join(self.save_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                model_files.append({
                    'filename': file,
                    'filepath': file_path,
                    'size_mb': file_size,
                    'modified_time': time.ctime(os.path.getmtime(file_path))
                })
        
        if model_files:
            self.logger.info(f"📁 保存目录: {self.save_dir}")
            self.logger.info("💾 已保存的模型文件:")
            for model in sorted(model_files, key=lambda x: x['filename']):
                self.logger.info(f"  - {model['filename']} ({model['size_mb']:.1f}MB, {model['modified_time']})")
        else:
            self.logger.info(f"📁 保存目录为空: {self.save_dir}")
        
        return model_files


def create_logger():
    """创建日志器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main(train_mode=False, test_mode=False, timestamp=None):
    """主函数"""
    # 创建日志器
    create_logger()
    
    # 训练参数
    diffusion_model_path = 'tb_logs/tsp_diffusion/version_0/checkpoints/last.ckpt'
    
    # 可选配置：控制是否使用额外的神经网络
    # network_type选项：'linear', 'mlp', 'none'
    use_neural_network = False
    network_type = 'none'  # 可以改为 'mlp', 'none' 来测试不同配置
    
    print(f"配置: use_neural_network={use_neural_network}, network_type={network_type}")
    print("网络类型说明:")
    print("  - 'linear': 简单线性层增强")
    print("  - 'mlp': 多层感知机增强") 
    print("  - 'none': 纯邻接矩阵决策")
    
    # 创建训练器
    trainer = DiffusionRLTrainer(
        diffusion_model_path=diffusion_model_path,
        num_nodes=50,
        batch_size=32,
        pomo_size=50,
        lr=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        # 新增参数
        use_neural_network=use_neural_network,
        network_type=network_type,
        embedding_dim=64,
        timestamp=timestamp
    )
    
    if train_mode:
        # 开始训练
        trainer.train(num_epochs=1000, log_interval=10)
    
    if test_mode:
        # 评估  
        print("\n" + "="*60)
        print("🔄 演示模型加载功能:")
        
        # 列出保存的模型
        saved_models = trainer.list_saved_models()
        
        # 加载最佳模型进行最终测试
        if trainer.load_best_model():
            print("✅ 成功加载最佳模型，进行最终测试...")
            test_length, _ = trainer.evaluate(num_test_instances=200, visualize_solutions=True, max_visualizations=8)
            print(f"🎯 使用最佳模型的测试结果: {test_length:.4f}")
        
        print("="*60)


if __name__ == "__main__":
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # timestamp = '20250606_224244'
    timestamp = None
    main(train_mode=True, test_mode=True, timestamp=timestamp) 