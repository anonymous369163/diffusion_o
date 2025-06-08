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
from logging import getLogger
import logging
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import time

# 导入扩散模型相关
sys.path.append('/root/code/difusco_o/difusco_o/DIFUSCO/difusco')
from difusco.utils.diffusion_schedulers import InferenceSchedule

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
        """重置环境状态"""
        self.points = points
        self.adj_matrix = adj_matrix
        
        device = points.device
        
        # 初始化状态
        self.visited_mask = torch.zeros(self.batch_size, self.pomo_size, self.num_nodes, dtype=torch.bool, device=device)
        self.current_node = torch.zeros(self.batch_size, self.pomo_size, dtype=torch.long, device=device)
        self.tour = [torch.zeros(self.batch_size, self.pomo_size, dtype=torch.long, device=device)]
        self.tour_length = torch.zeros(self.batch_size, self.pomo_size, device=device)
        self.done = torch.zeros(self.batch_size, self.pomo_size, dtype=torch.bool, device=device)
        
        # 确保索引在正确的设备上
        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(self.batch_size, self.pomo_size)
        
        # 标记起始节点为已访问
        new_visited_mask = self.visited_mask.clone()
        new_visited_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = True
        self.visited_mask = new_visited_mask
        
        return self.get_state()
    
    def step(self, action):
        """执行动作"""
        # 更新当前节点
        prev_node = self.current_node.clone()
        self.current_node = action
        
        # 计算移动距离（基于真实欧几里得距离）
        step_distance = self.calculate_distance(prev_node, self.current_node)
        self.tour_length += step_distance
        
        # 标记节点为已访问
        new_visited_mask = self.visited_mask.clone()
        new_visited_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = True
        self.visited_mask = new_visited_mask
        
        # 添加到路径
        self.tour.append(self.current_node.clone())
        
        # 检查是否完成
        if len(self.tour) == self.num_nodes:
            # 回到起始节点
            final_distance = self.calculate_distance(self.current_node, torch.zeros_like(self.current_node))
            self.tour_length += final_distance
            self.done = torch.ones_like(self.done)
            reward = -self.tour_length
        else:
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
        batch_size, pomo_size = node1.shape
        
        # 获取节点坐标
        node1_coords = self.points[self.BATCH_IDX, node1]
        node2_coords = self.points[self.BATCH_IDX, node2]
        
        # 计算欧几里得距离
        distance = torch.sqrt(((node1_coords - node2_coords) ** 2).sum(dim=-1))
        return distance


class TSPRLModel(nn.Module):
    """基于邻接矩阵的TSP强化学习模型"""
    
    def __init__(self, embedding_dim=64, use_linear_enhancement=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_linear_enhancement = use_linear_enhancement
        
        # 可选的线性层增强
        if self.use_linear_enhancement:
            self.node_embedding = nn.Linear(2, embedding_dim)
        
    def forward(self, state):
        """前向传播 - 以邻接矩阵为主要决策依据"""
        points = state['points']
        current_node = state['current_node']
        visited_mask = state['visited_mask']
        adj_matrix = state['adj_matrix']
        
        batch_size, pomo_size = current_node.shape
        num_nodes = points.shape[1]
        
        # 动态创建批次索引
        BATCH_IDX = torch.arange(batch_size, device=current_node.device)[:, None].expand(batch_size, pomo_size)
        
        # 主要决策依据：从邻接矩阵获取权重
        adj_logits = adj_matrix[BATCH_IDX, current_node]
        
        if self.use_linear_enhancement:
            # 线性层增强模式
            node_features = self.node_embedding(points)
            current_features = node_features[BATCH_IDX, current_node]
            
            # 计算特征相似度
            node_features_expanded = node_features.unsqueeze(1).expand(batch_size, pomo_size, num_nodes, self.embedding_dim)
            current_features_expanded = current_features.unsqueeze(2).expand(batch_size, pomo_size, num_nodes, self.embedding_dim)
            feature_similarity = (node_features_expanded * current_features_expanded).sum(dim=-1)
            
            # 结合邻接矩阵和特征相似度
            alpha = 0.8  # 邻接矩阵权重
            beta = 0.2   # 特征相似度权重
            logits = alpha * adj_logits + beta * feature_similarity
        else:
            # 纯邻接矩阵模式
            logits = adj_logits
        
        # 应用访问掩码
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
                 use_single_case=True,
                 use_linear_enhancement=True):
        
        self.device = device
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.pomo_size = pomo_size
        self.use_single_case = use_single_case
        self.use_linear_enhancement = use_linear_enhancement
        
        # 初始化扩散模型
        self.diffusion_model = self.load_diffusion_model(diffusion_model_path)
        
        # 如果使用单一案例，从测试集中获取固定案例
        if self.use_single_case:
            self.fixed_case = self.load_fixed_case()
            self.logger = getLogger(name='diffusion_rl_trainer')
            self.logger.info("使用单一案例训练模式")
            self.logger.info(f"固定案例节点数: {self.fixed_case['points'].shape[0]}")
            self.logger.info(f"固定案例最优长度: {self.fixed_case['gt_length']:.4f}")
        else:
            self.fixed_case = None
        
        # 初始化强化学习模型
        self.rl_model = TSPRLModel(embedding_dim=64, use_linear_enhancement=self.use_linear_enhancement).to(device)
        
        # 日志增强模式信息
        if not hasattr(self, 'logger'):
            self.logger = getLogger(name='diffusion_rl_trainer')
        
        if self.use_linear_enhancement:
            self.logger.info("使用线性层增强模式")
        else:
            self.logger.info("使用纯邻接矩阵模式")
        
        # 初始化环境
        self.env = TSPEnv(batch_size, num_nodes, pomo_size)
        
        # 初始化优化器
        all_params = list(self.diffusion_model.parameters()) + list(self.rl_model.parameters())
        self.optimizer = Optimizer(all_params, lr=lr)
        self.scheduler = Scheduler(self.optimizer, milestones=[100, 200], gamma=0.1)
        
        # 训练统计
        self.global_step = 0
        self.best_tour_length = float('inf')
        self.best_eval_length = float('inf')
        
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
            """生成邻接矩阵的Tensor版本，保持梯度连接"""
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).float().to(self.device)
            
            batch_size, num_nodes, _ = points.shape
            
            # 初始化随机噪声矩阵
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
            
            # 扩散迭代
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)
                
                t_tensor = torch.from_numpy(t1).view(1).to(self.device)
                
                if model.diffusion_type == 'gaussian':
                    epsilon_pred = model.forward(
                        points.float().to(self.device),
                        xt.float().to(self.device),
                        t_tensor.float().to(self.device),
                        None
                    )
                    epsilon_pred = epsilon_pred.squeeze(1)
                    xt = model.gaussian_posterior(t2, t_tensor, epsilon_pred, xt)
                else:
                    x0_pred = model.forward(
                        points.float().to(self.device),
                        xt.float().to(self.device),
                        t_tensor.float().to(self.device),
                        None
                    )
                    x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                    xt = model.categorical_posterior(t2, t_tensor, x0_pred_prob, xt)
            
            # 最终处理
            if model.diffusion_type == 'gaussian':
                adj_matrix = xt * 0.5 + 0.5
            else:
                adj_matrix = xt.float() + 1e-6
            
            return adj_matrix
        
        model.generate_adj_tensor = generate_adj_tensor
        return model
    
    def load_fixed_case(self):
        """从测试数据集中加载一个固定案例"""
        # 临时加载模型获取测试数据
        from difusco.difusion_tool import arg_parser, TSPModel_v2
        
        args = arg_parser()
        temp_model = TSPModel_v2(param_args=args)
        test_dataloader = temp_model.test_dataloader()
        
        # 获取第一个测试案例
        first_batch = next(iter(test_dataloader))
        
        # 解析数据
        if not temp_model.sparse:
            real_batch_idx, points, adj_matrix_gt, gt_tour = first_batch
            points = points[0]  # 取第一个实例
            gt_tour = gt_tour[0]
        else:
            real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = first_batch
            points = graph_data.x.reshape((-1, self.num_nodes, 2))[0]
            gt_tour = gt_tour.reshape(-1, self.num_nodes)[0]
        
        # 计算真实最优解的成本
        points_np = points.cpu().numpy()
        gt_tour_np = gt_tour.cpu().numpy()
        
        gt_cost = 0.0
        for i in range(len(gt_tour_np)):
            start_node = gt_tour_np[i]
            end_node = gt_tour_np[(i + 1) % len(gt_tour_np)]
            distance = np.sqrt(np.sum((points_np[start_node] - points_np[end_node]) ** 2))
            gt_cost += distance
        
        return {
            'points': points_np,
            'gt_tour': gt_tour_np,
            'gt_length': gt_cost
        }
    
    def generate_problems(self, batch_size):
        """生成TSP问题实例"""
        if self.use_single_case and self.fixed_case is not None:
            # 使用固定案例，复制到整个batch
            points = torch.from_numpy(self.fixed_case['points']).float().to(self.device)
            points = points.unsqueeze(0).repeat(batch_size, 1, 1)
            return points
        else:
            # 原来的随机生成逻辑
            points = torch.rand(batch_size, self.num_nodes, 2, device=self.device)
            return points
    
    def train_one_batch(self):
        """训练一个批次"""
        # 生成问题实例
        points = self.generate_problems(self.batch_size)
        
        # 使用扩散模型生成邻接矩阵
        adj_matrix = self.diffusion_model.generate_adj_tensor(points)
        
        # 重置环境
        state = self.env.reset(points, adj_matrix)
        
        # 收集轨迹
        log_probs = []
        rewards = []
        
        # POMO rollout
        for step in range(self.num_nodes - 1):
            logits = self.rl_model(state)
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
        
        # 计算advantage
        baseline = total_rewards.mean(dim=1, keepdim=True)
        advantage = total_rewards - baseline
        
        # 计算总log概率
        total_log_prob = torch.stack(log_probs).sum(dim=0)
        
        # 计算REINFORCE损失
        rl_loss = -(advantage * total_log_prob).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        rl_loss.backward()
        self.optimizer.step()
        
        # 计算统计信息
        best_rewards = total_rewards.max(dim=1)[0]
        avg_reward = best_rewards.mean().item()
        avg_tour_length = -avg_reward
        
        self.global_step += 1
        
        return avg_tour_length, rl_loss.item()
    
    def train(self, num_epochs=1000, log_interval=10):
        """训练主循环"""
        self.logger.info("开始训练扩散模型+强化学习...")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练模式
            self.rl_model.train()
            self.diffusion_model.train()
            
            # 训练一个批次
            avg_length, total_loss = self.train_one_batch()
            
            # 学习率调度
            self.scheduler.step()
            
            # 更新最佳记录
            if avg_length < self.best_tour_length:
                self.best_tour_length = avg_length
            
            # 记录日志
            if epoch % log_interval == 0:
                elapsed_time = time.time() - start_time
                self.logger.info(f"Epoch {epoch:4d}: "
                               f"平均路径长度={avg_length:.4f}, "
                               f"总损失={total_loss:.4f}, "
                               f"最佳长度={self.best_tour_length:.4f}, "
                               f"用时={elapsed_time:.1f}s")
            
            # 定期评估
            if epoch % 50 == 0:
                self.logger.info(f"进行第{epoch}轮评估...")
                # 在单一案例模式下，使用更小的评估集以加快评估速度
                eval_instances = 1 if self.use_single_case else 100
                eval_length, is_best = self.evaluate(num_test_instances=eval_instances, visualize_solutions=True, max_visualizations=3)
                self.logger.info(f"评估完成，平均路径长度: {eval_length:.4f}")
                
                # 如果是最佳结果，保存模型
                if is_best:
                    self.save_model(eval_length, epoch)
        
        # 训练结束
        total_time = time.time() - start_time
        self.logger.info(f"训练完成！总用时: {total_time:.1f}s")
        self.logger.info(f"最佳路径长度: {self.best_tour_length:.4f}")
        
        # 最终评估
        final_eval_length, _ = self.evaluate(num_test_instances=100, visualize_solutions=True, max_visualizations=5)
        self.logger.info(f"最终评估结果: {final_eval_length:.4f}")
    
    def evaluate(self, num_test_instances=100, visualize_solutions=True, max_visualizations=5):
        """评估模型性能 - 使用扩散模型的测试数据集"""
        self.logger.info("开始模型评估...")
        
        self.rl_model.eval()
        self.diffusion_model.eval()
        
        total_lengths = []
        total_gt_lengths = []
        solutions_to_visualize = []
        
        if self.use_single_case and self.fixed_case is not None:
            # 单一案例模式：只在固定案例上评估
            self.logger.info("单一案例评估模式")
            
            # 准备固定案例数据
            points = torch.from_numpy(self.fixed_case['points']).float().to(self.device)
            points = points.unsqueeze(0).repeat(self.batch_size, 1, 1)
            
            gt_tour = self.fixed_case['gt_tour']
            gt_length = self.fixed_case['gt_length']
            
            gt_lengths = [gt_length] * self.batch_size
            total_gt_lengths.extend(gt_lengths)
            
            with torch.no_grad():
                # 使用扩散模型生成邻接矩阵
                adj_matrix = self.diffusion_model.generate_adj_tensor(points)
                
                # 重置环境
                state = self.env.reset(points, adj_matrix)
                
                # 记录路径
                batch_tours = [[[] for _ in range(self.pomo_size)] for _ in range(self.batch_size)]
                for b in range(self.batch_size):
                    for p in range(self.pomo_size):
                        batch_tours[b][p].append(0)  # 起始节点
                
                # 贪心解码
                for step in range(self.num_nodes - 1):
                    logits = self.rl_model(state)
                    actions = logits.argmax(dim=-1)
                    
                    # 记录路径
                    for b in range(self.batch_size):
                        for p in range(self.pomo_size):
                            batch_tours[b][p].append(actions[b, p].item())
                    
                    state, reward, done = self.env.step(actions)
                    
                    if done.all():
                        break
                
                # 完成环路
                for b in range(self.batch_size):
                    for p in range(self.pomo_size):
                        batch_tours[b][p].append(0)  # 回到起始节点
                
                # 收集最优路径长度 (注意：这里是负的reward，需要取反)
                best_lengths = (-state['tour_length']).max(dim=1)[0]
                total_lengths.extend(best_lengths.cpu().numpy())
                
                # 保存解决方案用于可视化
                if visualize_solutions:
                    for b in range(min(self.batch_size, max_visualizations)):
                        best_pomo_idx = (-state['tour_length'][b]).argmax().item()
                        best_tour_length = (-state['tour_length'][b, best_pomo_idx]).item()
                        
                        solution_info = {
                            'points': points[b].cpu().numpy(),
                            'adj_matrix': adj_matrix[b].cpu().numpy(),
                            'tour_nodes': batch_tours[b][best_pomo_idx],
                            'tour_length': best_tour_length,
                            'gt_tour': gt_tour,
                            'gt_length': gt_length,
                        }
                        solutions_to_visualize.append(solution_info)
                        
        else:
            # 原来的多案例评估逻辑
            # 获取测试数据加载器
            test_dataloader = self.diffusion_model.test_dataloader()
            self.logger.info(f"测试数据集大小: {len(test_dataloader)}")
            
            # 限制测试实例数量
            max_batches = min(num_test_instances // self.batch_size, len(test_dataloader))
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    if batch_idx >= max_batches:
                        break
                    
                # 解析测试数据（参考difusion_tool.py中的格式）
                if not self.diffusion_model.sparse:
                    real_batch_idx, points, adj_matrix_gt, gt_tour = batch
                    # 移动到正确的设备
                    points = points.to(self.device)
                    gt_tour = gt_tour.to(self.device)
                    
                    # 只取第一个实例（测试时batch_size通常为1）
                    if points.shape[0] == 1:
                        points = points.repeat(self.batch_size, 1, 1)
                        gt_tour = gt_tour.repeat(self.batch_size, 1)
                    else:
                        # 如果batch大小不匹配，截取或填充
                        if points.shape[0] < self.batch_size:
                            points = points[:1].repeat(self.batch_size, 1, 1)
                            gt_tour = gt_tour[:1].repeat(self.batch_size, 1)
                        else:
                            points = points[:self.batch_size]
                            gt_tour = gt_tour[:self.batch_size]
                else:
                    # 处理稀疏格式（如果需要）
                    real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
                    points = graph_data.x.reshape((-1, self.num_nodes, 2))
                    gt_tour = gt_tour.reshape(-1, self.num_nodes)
                    
                    # 移动到正确的设备并处理batch大小
                    points = points.to(self.device)
                    gt_tour = gt_tour.to(self.device)
                    
                    if points.shape[0] < self.batch_size:
                        points = points[:1].repeat(self.batch_size, 1, 1)
                        gt_tour = gt_tour[:1].repeat(self.batch_size, 1)
                    else:
                        points = points[:self.batch_size]
                        gt_tour = gt_tour[:self.batch_size]
                
                # 计算真实最优解的成本
                gt_lengths = []
                for b in range(points.shape[0]):
                    gt_tour_b = gt_tour[b].cpu().numpy()
                    points_b = points[b].cpu().numpy()
                    
                    # 计算真实最优路径的成本
                    gt_cost = 0.0
                    for i in range(len(gt_tour_b)):
                        start_node = gt_tour_b[i]
                        end_node = gt_tour_b[(i + 1) % len(gt_tour_b)]
                        distance = np.sqrt(np.sum((points_b[start_node] - points_b[end_node]) ** 2))
                        gt_cost += distance
                    gt_lengths.append(gt_cost)
                
                total_gt_lengths.extend(gt_lengths)
                
                # 使用扩散模型生成邻接矩阵
                adj_matrix = self.diffusion_model.generate_adj_tensor(points)
                
                # 重置环境
                state = self.env.reset(points, adj_matrix)
                
                # 记录路径
                batch_tours = [[[] for _ in range(self.pomo_size)] for _ in range(self.batch_size)]
                for b in range(self.batch_size):
                    for p in range(self.pomo_size):
                        batch_tours[b][p].append(0)  # 起始节点
                
                # 贪心解码
                for step in range(self.num_nodes - 1):
                    logits = self.rl_model(state)
                    actions = logits.argmax(dim=-1)
                    
                    # 记录路径
                    for b in range(self.batch_size):
                        for p in range(self.pomo_size):
                            batch_tours[b][p].append(actions[b, p].item())
                    
                    state, reward, done = self.env.step(actions)
                    
                    if done.all():
                        break
                
                # 完成环路
                for b in range(self.batch_size):
                    for p in range(self.pomo_size):
                        batch_tours[b][p].append(0)  # 回到起始节点
                
                # 收集最优路径长度 (注意：这里是负的reward，需要取反)
                best_lengths = (-state['tour_length']).max(dim=1)[0]
                total_lengths.extend(best_lengths.cpu().numpy())
                
                # 保存解决方案用于可视化
                if visualize_solutions and len(solutions_to_visualize) < max_visualizations:
                    for b in range(min(self.batch_size, max_visualizations - len(solutions_to_visualize))):
                        best_pomo_idx = (-state['tour_length'][b]).argmax().item()
                        best_tour_length = (-state['tour_length'][b, best_pomo_idx]).item()
                        
                        solution_info = {
                            'points': points[b].cpu().numpy(),
                            'adj_matrix': adj_matrix[b].cpu().numpy(),
                            'tour_nodes': batch_tours[b][best_pomo_idx],
                            'tour_length': best_tour_length,
                            'gt_tour': gt_tour[b].cpu().numpy(),
                            'gt_length': gt_lengths[b],
                        }
                        solutions_to_visualize.append(solution_info)
        
        # 计算统计信息
        avg_length = np.mean(total_lengths)
        std_length = np.std(total_lengths)
        min_length = np.min(total_lengths)
        max_length = np.max(total_lengths)
        
        # 计算与最优解的比较
        avg_gt_length = np.mean(total_gt_lengths)
        gap_values = [(pred - gt) / gt * 100 for pred, gt in zip(total_lengths, total_gt_lengths)]
        avg_gap = np.mean(gap_values)
        std_gap = np.std(gap_values)
        
        # 检查是否为最佳结果
        is_best = avg_length < self.best_eval_length
        if is_best:
            self.best_eval_length = avg_length
            self.logger.info(f"发现新的最佳评估结果! 平均路径长度: {avg_length:.4f}")
        
        # 可视化解决方案
        if visualize_solutions and solutions_to_visualize:
            self.logger.info(f"正在可视化 {len(solutions_to_visualize)} 个评估解决方案...")
            self.visualize_solutions(solutions_to_visualize, avg_length, is_best)
        
        self.logger.info(f"评估结果 - 预测平均长度: {avg_length:.4f} ± {std_length:.4f}")
        self.logger.info(f"评估结果 - 最优平均长度: {avg_gt_length:.4f}")
        self.logger.info(f"评估结果 - 相对差距: {avg_gap:.2f}% ± {std_gap:.2f}%")
        self.logger.info(f"评估结果 - 长度范围: [{min_length:.4f}, {max_length:.4f}]")
        
        return avg_length, is_best
    
    def visualize_solutions(self, solutions, avg_length, is_best):
        """可视化评估过程中的解决方案"""
        try:
            import matplotlib.pyplot as plt
            
            # 找到最佳和最差的解决方案
            best_solution = min(solutions, key=lambda x: x['tour_length'])
            worst_solution = max(solutions, key=lambda x: x['tour_length'])
            
            # 创建多子图展示
            n_solutions = min(len(solutions), 4)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            solutions_to_show = [best_solution, worst_solution] + solutions[:2] if len(solutions) > 2 else solutions
            
            for i, solution in enumerate(solutions_to_show[:n_solutions]):
                ax = axes[i]
                
                points_np = solution['points']
                tour_nodes = solution['tour_nodes']
                tour_length = solution['tour_length']
                
                # 获取真实最优解信息
                gt_tour = solution.get('gt_tour', None)
                gt_length = solution.get('gt_length', None)
                
                # 绘制节点
                ax.scatter(points_np[:, 0], points_np[:, 1], c='red', s=60, zorder=3)
                
                # 标注节点编号
                for j, (x, y) in enumerate(points_np):
                    ax.annotate(str(j), (x, y), xytext=(3, 3), textcoords='offset points', 
                              fontsize=7, fontweight='bold')
                
                # 绘制真实最优路径（如果存在）
                if gt_tour is not None:
                    gt_tour_coords = points_np[gt_tour]
                    
                    # 绘制最优路径线段（绿色虚线）
                    for j in range(len(gt_tour)):
                        start_point = gt_tour_coords[j]
                        end_point = gt_tour_coords[(j + 1) % len(gt_tour)]
                        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                               'g--', linewidth=2, alpha=0.8, zorder=1, label='最优解' if j == 0 else "")
                
                # 绘制预测的TSP路径（蓝色实线）
                tour_coords = points_np[tour_nodes]
                
                # 绘制路径线段
                for j in range(len(tour_nodes) - 1):
                    start_point = tour_coords[j]
                    end_point = tour_coords[j + 1]
                    ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                           'b-', linewidth=1.5, alpha=0.7, zorder=2, label='预测解' if j == 0 else "")
                
                # 高亮起始节点
                start_point = points_np[0]
                circle = plt.Circle((start_point[0], start_point[1]), 0.025, 
                                  color='orange', fill=True, zorder=4)
                ax.add_patch(circle)
                
                # 设置标题
                if i == 0 and solution == best_solution:
                    title_prefix = "🏆 最佳解"
                elif i == 1 and solution == worst_solution:
                    title_prefix = "📉 最差解"
                else:
                    title_prefix = f"解决方案 #{i+1}"
                
                # 计算相对差距
                if gt_length is not None:
                    gap = (tour_length - gt_length) / gt_length * 100
                    title_text = f'{title_prefix}\n预测: {tour_length:.3f} | 最优: {gt_length:.3f}\n差距: {gap:.2f}%'
                else:
                    title_text = f'{title_prefix}\n长度: {tour_length:.3f}'
                    
                ax.set_title(title_text, fontsize=10)
                ax.set_xlabel('X坐标', fontsize=8)
                ax.set_ylabel('Y坐标', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                # 添加图例（仅为第一个子图）
                if i == 0 and gt_tour is not None:
                    ax.legend(loc='upper right', fontsize=8)
            
            # 隐藏未使用的子图
            for i in range(n_solutions, 4):
                axes[i].set_visible(False)
            
            # 计算总体差距信息
            avg_gt_length = np.mean([s.get('gt_length', 0) for s in solutions if s.get('gt_length') is not None])
            if avg_gt_length > 0:
                overall_gap = (avg_length - avg_gt_length) / avg_gt_length * 100
                gap_info = f" | 平均差距: {overall_gap:.2f}%"
            else:
                gap_info = ""
            
            # 设置总标题
            best_indicator = " 🏆 新最佳!" if is_best else ""
            fig.suptitle(f'评估解决方案可视化{best_indicator}\n平均长度: {avg_length:.4f}{gap_info} | 步骤: {self.global_step}', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图片
            os.makedirs('results', exist_ok=True)
            plt.savefig(f'results/solutions_step_{self.global_step}.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"保存可视化结果到: results/solutions_step_{self.global_step}.png")
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.warning(f"可视化失败: {e}")
    
    def save_model(self, eval_length, epoch):
        """保存最佳模型"""
        os.makedirs('saved_models', exist_ok=True)
        
        model_data = {
            'epoch': epoch,
            'eval_length': eval_length,
            'rl_model_state_dict': self.rl_model.state_dict(),
            'diffusion_model_state_dict': self.diffusion_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        model_path = 'saved_models/best_model.pt'
        torch.save(model_data, model_path)
        self.logger.info(f"保存最佳模型: {model_path} (评估长度: {eval_length:.4f})")


def create_logger():
    """创建日志器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """主函数"""
    create_logger()
    
    # 训练参数
    diffusion_model_path = 'tb_logs/tsp_diffusion/version_0/checkpoints/last.ckpt'
    
    # 创建训练器 - 启用单一案例训练模式
    trainer = DiffusionRLTrainer(
        diffusion_model_path=diffusion_model_path,
        num_nodes=50,
        batch_size=8,  # 减小batch size以便观察单一案例的训练效果
        pomo_size=50,
        lr=1e-3,  # 提高学习率以加快在单一案例上的收敛
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_single_case=True,  # 明确启用单一案例模式
        use_linear_enhancement=False  # 线性层增强模式：True=启用，False=纯邻接矩阵模式
    )
    
    # 在训练前先评估一次，看看初始性能
    trainer.logger.info("=== 训练前初始评估 ===")
    initial_eval_length, _ = trainer.evaluate(num_test_instances=1, visualize_solutions=True, max_visualizations=5)
    trainer.logger.info(f"初始评估结果: {initial_eval_length:.4f}")
    
    # 开始训练 - 使用更频繁的日志和评估
    trainer.train(num_epochs=200, log_interval=5)  # 更频繁的日志记录
    
    # 训练后最终评估
    trainer.logger.info("=== 训练后最终评估 ===")
    final_eval_length, _ = trainer.evaluate(num_test_instances=1, visualize_solutions=True, max_visualizations=5)
    trainer.logger.info(f"最终评估结果: {final_eval_length:.4f}")
    trainer.logger.info(f"相对改进: {((initial_eval_length - final_eval_length) / initial_eval_length * 100):.2f}%")


if __name__ == "__main__":
    main() 