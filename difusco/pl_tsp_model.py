"""Lightning module for training the DIFUSCO TSP model.

数值稳定性配置参数说明:
- rl_enable: 是否启用强化学习损失 (默认: True)
- rl_skip_on_error: 遇到错误时是否跳过而不是崩溃 (默认: True)
- rl_max_failures: 最大失败次数，超过后自动禁用 (默认: 10)
- max_logit_value: 最大logit值，防止softmax溢出 (默认: 50.0)
- min_prob_value: 最小概率值，防止log(0)错误 (默认: 1e-8)
- max_advantage: 最大优势值，防止梯度爆炸 (默认: 20.0)
- pomo_temperature: POMO求解器温度，建议范围[0.1, 2.0] (默认: 1.0)
- rl_debug: 是否启用强化学习调试模式，输出详细诊断信息 (默认: False)
- test_debug: 是否启用测试调试模式，测试时输出详细诊断信息 (默认: False)

使用示例:
  args.rl_enable = True
  args.rl_skip_on_error = True  
  args.rl_max_failures = 10
  args.max_logit_value = 50.0
  args.min_prob_value = 1e-8
  args.max_advantage = 20.0
  args.pomo_temperature = 1.0
  args.rl_debug = False  # 设为True启用调试模式
  args.test_debug = False  # 设为True启用测试调试模式

调试模式功能:
- 当rl_debug=True时，会输出详细的数值诊断信息
- 包括邻接矩阵统计、掩码分析、softmax问题定位等
- 帮助快速定位nan/inf问题的具体原因
- 建议在遇到数值问题时临时启用进行调试

常见问题诊断:
1. 如果看到"所有选择都被屏蔽"：
   - 检查VRP约束是否过于严格
   - 查看载重、时间窗、长度限制等约束状态
   - 可能需要调整约束参数或初始化值

2. 如果看到"softmax后包含nan/inf"：
   - 通常是logits数值范围过大导致
   - 检查扩散模型输出是否合理
   - 可能需要调整max_logit_value参数

3. 如果看到"multinomial采样失败"：
   - 概率分布不合法（不归一化或包含负值）
   - 会自动回退到贪婪选择继续运行
"""

import os
from pickletools import pystring
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from co_datasets.tsp_graph_dataset import TSPGraphDataset, VRPGraphDataset
from co_datasets.hybrid_graph_dataset import HybridGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator  # , batched_two_opt_torch, merge_tours  # 注释掉因为改用强化学习方法


class HybridBatchSampler(torch.utils.data.BatchSampler):
    """
    自定义的批次采样器，确保每个批次内的问题类型一致
    专门用于HybridGraphDataset
    """
    
    def __init__(self, sampler, batch_size, drop_last=False):
        """
        初始化混合批次采样器
        
        Args:
            sampler: 基础采样器或HybridGraphDataset实例
            batch_size: 批次大小
            drop_last: 是否丢弃最后一个不完整的批次
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._shuffle = True  # 默认打乱批次顺序
        
        # 尝试多种方式获取数据集
        self.dataset = None
        self.sampler = None
        
        # 方案1: 传入的是HybridGraphDataset实例
        if hasattr(sampler, 'batch_mappings'):
            self.dataset = sampler
            self.sampler = torch.utils.data.SequentialSampler(sampler)
        # 方案2: 传入的是标准sampler，尝试从data_source获取
        elif hasattr(sampler, 'data_source'):
            self.sampler = sampler
            self.dataset = sampler.data_source
        # 方案3: 尝试从dataset属性获取
        elif hasattr(sampler, 'dataset'):
            self.sampler = sampler
            self.dataset = sampler.dataset
        # 方案4: 尝试从_dataset属性获取
        elif hasattr(sampler, '_dataset'):
            self.sampler = sampler
            self.dataset = sampler._dataset
        # 方案5: 检查sampler是否就是一个数据集（没有明显的sampler属性）
        elif hasattr(sampler, '__len__') and hasattr(sampler, '__getitem__'):
            # 这可能是一个数据集
            self.dataset = sampler
            self.sampler = torch.utils.data.SequentialSampler(sampler)
        else:
            # 调试信息：打印sampler的属性
            sampler_attrs = [attr for attr in dir(sampler) if not attr.startswith('_')]
            print(f"DEBUG: sampler类型: {type(sampler)}")
            print(f"DEBUG: sampler属性: {sampler_attrs[:10]}...")  # 只显示前10个属性
            
            # 尝试其他可能的属性名
            possible_dataset_attrs = ['data', 'source', 'base_dataset', 'wrapped_dataset']
            found_dataset = False
            for attr_name in possible_dataset_attrs:
                if hasattr(sampler, attr_name):
                    potential_dataset = getattr(sampler, attr_name)
                    if hasattr(potential_dataset, 'batch_mappings'):
                        print(f"DEBUG: 从{attr_name}属性找到HybridGraphDataset")
                        self.dataset = potential_dataset
                        self.sampler = torch.utils.data.SequentialSampler(potential_dataset)
                        found_dataset = True
                        break
            
            if not found_dataset:
                # 如果这是在分布式环境中的重新实例化，提供一个警告并尝试回退
                print(f"WARNING: 无法从sampler获取数据集，sampler类型: {type(sampler)}")
                print("这可能是分布式训练时的正常情况，将尝试禁用HybridBatchSampler功能")
                
                # 回退：创建一个基本的行为，但记录这种情况
                self.sampler = sampler
                self.dataset = None  # 标记为无法获取数据集
                self._fallback_mode = True
                return
        
        # 如果成功获取到数据集，进行验证
        if self.dataset is not None:
            # 检查数据集是否为HybridGraphDataset
            if not isinstance(self.dataset, HybridGraphDataset):
                print(f"WARNING: 数据集类型不是HybridGraphDataset: {type(self.dataset)}")
                # 不抛出错误，而是标记为fallback模式
                self._fallback_mode = True
                return
            
            # 确保数据集已经创建了批次映射
            if not hasattr(self.dataset, 'batch_mappings') or not self.dataset.batch_mappings:
                print("WARNING: 数据集未正确创建批次映射，将尝试重新创建")
                try:
                    # 尝试重新创建批次映射（如果数据集支持）
                    if hasattr(self.dataset, 'create_batch_mappings'):
                        self.dataset.create_batch_mappings()
                    else:
                        print("ERROR: 无法重新创建批次映射")
                        self._fallback_mode = True
                        return
                except Exception as e:
                    print(f"ERROR: 重新创建批次映射失败: {e}")
                    self._fallback_mode = True
                    return
        
        # 标记正常模式
        self._fallback_mode = False
    
    def set_shuffle(self, shuffle):
        """设置是否打乱批次顺序"""
        self._shuffle = shuffle
        
        # 在fallback模式下，记录但不执行特殊的shuffle操作
        if getattr(self, '_fallback_mode', False):
            print(f"WARNING: fallback模式下设置shuffle={shuffle}，但可能无法生效")
    
    def __iter__(self):
        """返回批次索引的迭代器"""
        # 如果是fallback模式，使用标准的批次采样逻辑
        if getattr(self, '_fallback_mode', False):
            print("WARNING: 使用fallback批次采样模式")
            # 使用基础sampler的逻辑
            if hasattr(self.sampler, '__iter__'):
                sampler_iter = iter(self.sampler)
                batch = []
                for idx in sampler_iter:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                # 处理最后一个不完整的批次
                if batch and not self.drop_last:
                    yield batch
            else:
                # 如果sampler没有迭代器，创建一个简单的顺序批次
                total_size = len(self.sampler) if hasattr(self.sampler, '__len__') else 0
                for start_idx in range(0, total_size, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, total_size)
                    if end_idx - start_idx == self.batch_size or not self.drop_last:
                        yield list(range(start_idx, end_idx))
            return
        
        # 正常模式：使用HybridGraphDataset的批次映射
        if self.dataset is None:
            print("ERROR: dataset为None，无法执行批次采样")
            return
        
        # 如果需要打乱，重新创建批次映射
        if self._shuffle:
            if hasattr(self.dataset, 'shuffle_batches'):
                self.dataset.shuffle_batches()
        
        # 返回所有批次的索引
        if hasattr(self.dataset, 'batch_mappings'):
            for batch_idx in range(len(self.dataset.batch_mappings)):
                batch_indices = self.dataset.batch_mappings[batch_idx]
                yield batch_indices
        else:
            print("ERROR: dataset没有batch_mappings属性")
    
    def __len__(self):
        """返回批次总数"""
        # 如果是fallback模式，计算基本的批次数量
        if getattr(self, '_fallback_mode', False):
            if hasattr(self.sampler, '__len__'):
                total_size = len(self.sampler)
                if self.drop_last:
                    return total_size // self.batch_size
                else:
                    return (total_size + self.batch_size - 1) // self.batch_size
            else:
                # 如果无法确定大小，返回一个默认值
                print("WARNING: 无法确定sampler的大小，返回默认批次数量")
                return 1
        
        # 正常模式：使用HybridGraphDataset的批次映射
        if self.dataset is not None and hasattr(self.dataset, 'batch_mappings'):
            return len(self.dataset.batch_mappings)
        else:
            print("WARNING: 无法确定批次数量，返回默认值")
            return 1


class HybridCollateFunction:
    """
    专门用于HybridGraphDataset的collate函数
    确保批次内问题类型一致并正确处理变长数据
    """
    
    def __init__(self):
        pass
    
    def __call__(self, batch):
        """
        处理批次数据，确保问题类型一致并处理变长数据
        
        Args:
            batch: 来自HybridGraphDataset的样本列表
            
        Returns:
            处理后的批次数据
        """
        try:
            # 检查批次内的问题类型
            problem_types = [item[4] for item in batch]
            unique_types = set(problem_types)
            
            if len(unique_types) > 1:
                print(f"警告：批次中包含多种问题类型：{unique_types}，使用第一个类型")
            
            batch_problem_type = problem_types[0]
            
            # 解包数据
            sample_indices, points, adj_matrices, tours, _ = zip(*batch)
            
            # 处理变长数据的填充
            max_nodes = max(p.shape[0] for p in points)
            max_tour_length = max(t.shape[0] for t in tours)
            
            # 填充points到相同大小
            padded_points = []
            for p in points:
                if p.shape[0] < max_nodes:
                    pad_size = max_nodes - p.shape[0]
                    padding = torch.zeros(pad_size, p.shape[1], dtype=p.dtype)
                    padded_p = torch.cat([p, padding], dim=0)
                else:
                    padded_p = p
                padded_points.append(padded_p)
            
            # 填充adj_matrices到相同大小
            padded_adj_matrices = []
            for adj in adj_matrices:
                if adj.shape[0] < max_nodes:
                    pad_size = max_nodes - adj.shape[0]
                    padded_adj = torch.zeros(max_nodes, max_nodes, dtype=adj.dtype)
                    padded_adj[:adj.shape[0], :adj.shape[1]] = adj
                else:
                    padded_adj = adj
                padded_adj_matrices.append(padded_adj)
            
            # 填充tours到相同长度
            padded_tours = []
            for t in tours:
                if t.shape[0] < max_tour_length:
                    pad_size = max_tour_length - t.shape[0]
                    padding = torch.full((pad_size,), -1, dtype=t.dtype)
                    padded_t = torch.cat([t, padding], dim=0)
                else:
                    padded_t = t
                padded_tours.append(padded_t)
            
            # 堆叠tensor
            sample_indices = torch.stack(sample_indices)
            points = torch.stack(padded_points)
            adj_matrices = torch.stack(padded_adj_matrices)
            tours = torch.stack(padded_tours)
            
            return sample_indices, points, adj_matrices, tours, batch_problem_type
            
        except Exception as e:
            print(f"ERROR in HybridCollateFunction: {e}")
            print(f"Batch info - size: {len(batch)}")
            if len(batch) > 0:
                print(f"First item info - type: {type(batch[0])}, length: {len(batch[0])}")
                if hasattr(batch[0], '__iter__'):
                    for i, item in enumerate(batch[0]):
                        if hasattr(item, 'shape'):
                            print(f"  Item {i}: shape {item.shape}, dtype {item.dtype}")
                        else:
                            print(f"  Item {i}: type {type(item)}, value {item}")
            raise


def simulate_vrp_execution(tour, points_with_features, problem_type="TSP"):
    """
    精确模拟VRP路径执行过程，使用与求解器相同的状态更新逻辑
    Args:
        tour: list or np.array - 路径序列
        points_with_features: np.array of shape (num_nodes, 7) - 节点特征
        problem_type: str - 问题类型
    Returns:
        execution_history: dict - 包含每一步的状态信息
    """
    # 确定问题属性
    attribute_c = 'C' in problem_type or problem_type in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW", "OVRPB", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]
    attribute_tw = 'TW' in problem_type
    attribute_o = 'O' in problem_type and problem_type.startswith('O')
    attribute_b = 'B' in problem_type  
    attribute_l = 'L' in problem_type
    
    round_error_epsilon = 0.000001
    
    # 提取特征
    if points_with_features is not None and points_with_features.shape[-1] >= 7:
        coords = points_with_features[:, :2]  # x, y坐标
        demands = points_with_features[:, 2]  # 需求
        early_tw = points_with_features[:, 3]  # 早期时间窗
        late_tw = points_with_features[:, 4]  # 晚期时间窗
        route_open = points_with_features[:, 5]  # 开放路径标志
        length_limit = points_with_features[:, 6]  # 路径长度限制
    else:
        # 使用默认值
        num_nodes = len(points_with_features) if points_with_features is not None else len(tour)
        coords = points_with_features[:, :2] if points_with_features is not None else np.random.rand(num_nodes, 2)
        demands = np.concatenate([np.array([0]), np.random.rand(num_nodes-1) * 0.1])
        early_tw = np.zeros(num_nodes)
        late_tw = np.zeros(num_nodes)
        route_open = np.zeros(num_nodes)
        length_limit = np.full(num_nodes, 3.0)
    
    # 初始化状态
    current_load = 1.0  # 当前载重，初始为满容量
    current_time = 0.0  # 当前时间
    # 修复：路径长度限制是常量，使用客户节点的值（避免depot的0值）
    if len(length_limit) > 1:
        initial_length = length_limit[1]  # 使用第一个客户节点的长度限制
    else:
        initial_length = 3.0  # 默认值
    remaining_length = initial_length  # 剩余路径长度
    
    # 约束违反统计
    constraint_violations = {
        'capacity_violations': 0,
        'time_window_violations': 0,
        'length_violations': 0,
        'total_violations': 0
    }
    
    # 执行历史记录
    execution_history = {
        'nodes': [],           # 访问的节点序列
        'loads': [],           # 载重变化
        'times': [],           # 时间变化  
        'lengths': [],         # 剩余长度变化
        'distances': [],       # 累积距离
        'violations': [],      # 违反约束的步骤
        'route_segments': [],  # 路径段信息
        'at_depot': [],        # 是否在depot
        'constraint_violations': constraint_violations
    }
    
    # 修复：分别跟踪总累积距离和当前路径段距离
    total_traveled_distance = 0.0  # 总累积距离（用于记录）
    current_segment_distance = 0.0  # 当前路径段累积距离（用于长度约束检查）
    current_route = []
    route_segments = []
    
    # 记录初始状态
    execution_history['nodes'].append(tour[0])
    execution_history['loads'].append(current_load)
    execution_history['times'].append(current_time)
    execution_history['lengths'].append(remaining_length)
    execution_history['distances'].append(total_traveled_distance)
    execution_history['at_depot'].append(tour[0] == 0)
    execution_history['violations'].append([])
    
    # 模拟路径执行
    for i in range(len(tour) - 1):
        current_node = tour[i]
        next_node = tour[i + 1]
        
        current_route.append(current_node)
        
        # 计算移动距离（使用欧几里得距离）
        current_point = coords[current_node]
        next_point = coords[next_node]
        segment_distance = np.sqrt(np.sum((next_point - current_point) ** 2))
        
        # 检查约束并更新状态
        at_depot_now = (next_node == 0)
        step_violations = []
        
        # 1. 容量约束检查和更新
        if attribute_c:
            # 检查容量违反（在消耗需求之前）
            if current_load + round_error_epsilon < demands[next_node] and not at_depot_now:
                constraint_violations['capacity_violations'] += 1
                step_violations.append(('capacity', next_node, current_load, demands[next_node]))
            
            # 更新载重
            current_load -= demands[next_node]
            if at_depot_now:
                current_load = 1.0  # 在depot重置载重
        
        # 2. 时间窗约束检查和更新
        if attribute_tw:
            # 计算到达时间
            arrival_time = current_time + segment_distance
            
            # 检查时间窗违反
            if late_tw[next_node] > 0 and arrival_time > late_tw[next_node]:
                constraint_violations['time_window_violations'] += 1
                step_violations.append(('time_window', next_node, arrival_time, late_tw[next_node]))
            
            # 更新时间（考虑等待早期时间窗）
            current_time = max(arrival_time, early_tw[next_node])
            if at_depot_now:
                current_time = 0.0  # 在depot重置时间
        else:
            current_time += segment_distance
        
        # 3. 路径长度约束检查和更新
        if attribute_l:
            # 修复：路径长度约束应该基于当前路径段的累积距离
            # 检查长度违反：当前路径段距离 + 新的segment距离 是否超过剩余长度
            projected_segment_distance = current_segment_distance + segment_distance
            
            if attribute_o:
                # 开放路径：只需要检查到达目标节点的距离
                if remaining_length - round_error_epsilon < segment_distance:
                    constraint_violations['length_violations'] += 1
                    step_violations.append(('length', next_node, remaining_length, segment_distance))
            else:
                # 封闭路径：需要考虑返回depot的距离
                if not at_depot_now:
                    # 如果不是回到depot，需要考虑从目标节点返回depot的距离
                    return_to_depot_distance = np.sqrt(np.sum((coords[0] - next_point) ** 2))
                    total_distance_needed = segment_distance + return_to_depot_distance
                    if remaining_length - round_error_epsilon < total_distance_needed:
                        constraint_violations['length_violations'] += 1
                        step_violations.append(('length', next_node, remaining_length, total_distance_needed))
                else:
                    # 回到depot，只检查segment距离
                    if remaining_length - round_error_epsilon < segment_distance:
                        constraint_violations['length_violations'] += 1
                        step_violations.append(('length', next_node, remaining_length, segment_distance))
            
            # 更新剩余长度和当前路径段距离
            remaining_length -= segment_distance
            current_segment_distance += segment_distance
            
            # 关键修复：当回到depot时，重置路径段距离和长度限制
            if at_depot_now:
                current_segment_distance = 0.0  # 重置当前路径段距离
                # 重置为初始的路径长度限制（常量）
                remaining_length = initial_length
        
        # 更新累积距离
        total_traveled_distance += segment_distance
        
        # 记录当前步骤状态
        execution_history['nodes'].append(next_node)
        execution_history['loads'].append(current_load)
        execution_history['times'].append(current_time)
        execution_history['lengths'].append(remaining_length)
        execution_history['distances'].append(total_traveled_distance)
        execution_history['at_depot'].append(at_depot_now)
        execution_history['violations'].append(step_violations)
        
        # 检查是否回到depot（新路径段开始）
        if at_depot_now and i < len(tour) - 2:
            current_route.append(next_node)
            route_segments.append(current_route.copy())
            current_route = []
    
    # 添加最后一个路径段
    if current_route:
        route_segments.append(current_route)
    
    execution_history['route_segments'] = route_segments
    constraint_violations['total_violations'] = sum([
        constraint_violations['capacity_violations'],
        constraint_violations['time_window_violations'], 
        constraint_violations['length_violations']
    ])
    
    return execution_history


def visualize_vrp_solution(points, tour, points_with_features=None, problem_type="TSP", 
                          gt_tour=None, gt_cost=None, pred_cost=None, save_path=None, 
                          title_suffix="", show_constraints=True, figsize=(15, 10)):
    """
    可视化VRP问题的解决方案，包括精确的车辆状态、约束验证等信息
    
    Args:
        points: np.array of shape (num_nodes, 2) - 节点坐标
        tour: list or np.array - 预测的路径序列
        points_with_features: np.array of shape (num_nodes, 7) - 节点特征 [x, y, demand, early_tw, late_tw, route_open, length_limit]
        problem_type: str - 问题类型
        gt_tour: list or np.array - 真实路径（可选）
        gt_cost: float - 真实路径成本（可选）
        pred_cost: float - 预测路径成本（可选）
        save_path: str - 保存路径（可选）
        title_suffix: str - 标题后缀
        show_constraints: bool - 是否显示约束信息
        figsize: tuple - 图片大小
    
    Returns:
        fig: matplotlib figure对象
        execution_history: dict - 路径执行历史和约束违反情况
    """
    
    # 确定问题属性
    attribute_c = 'C' in problem_type or problem_type in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW", "OVRPB", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]
    attribute_tw = 'TW' in problem_type
    attribute_o = 'O' in problem_type and problem_type.startswith('O')
    attribute_b = 'B' in problem_type  
    attribute_l = 'L' in problem_type
    
    # 转换为numpy数组
    if isinstance(tour, torch.Tensor):
        tour = tour.cpu().numpy()
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if points_with_features is not None and isinstance(points_with_features, torch.Tensor):
        points_with_features = points_with_features.cpu().numpy()
    
    # 使用精确的状态模拟
    execution_history = simulate_vrp_execution(tour, points_with_features, problem_type)
    constraint_violations = execution_history['constraint_violations']
    
    # 创建子图布局
    if show_constraints and points_with_features is not None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 2, 1])
        ax_main = fig.add_subplot(gs[0, :2])  # 主路径图
        ax_info = fig.add_subplot(gs[0, 2])   # 约束信息
        ax_load = fig.add_subplot(gs[1, 0])   # 载重变化
        ax_time = fig.add_subplot(gs[1, 1])   # 时间变化
        ax_length = fig.add_subplot(gs[1, 2]) # 路径长度变化
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(10, 8))
        ax_info = ax_load = ax_time = ax_length = None
    
    # 主路径可视化
    num_nodes = len(points)
    
    # 绘制节点
    if problem_type != "TSP":
        # VRP问题：区分depot和客户节点
        depot_point = points[0]
        customer_points = points[1:]
        
        # 绘制depot（红色大方形）
        ax_main.scatter(depot_point[0], depot_point[1], c='red', s=200, marker='s', 
                       label='Depot', edgecolors='black', linewidths=2, zorder=5)
        
        # 绘制客户节点（蓝色圆）
        ax_main.scatter(customer_points[:, 0], customer_points[:, 1], c='lightblue', 
                       s=100, marker='o', label='Customers', edgecolors='blue', 
                       linewidths=1, zorder=3)
        
        # 添加节点编号
        for i in range(num_nodes):
            ax_main.annotate(str(i), (points[i, 0], points[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, fontweight='bold', zorder=6)
    else:
        # TSP问题：所有节点相同
        ax_main.scatter(points[:, 0], points[:, 1], c='lightblue', s=100, 
                       marker='o', edgecolors='blue', linewidths=1, zorder=3)
        for i in range(num_nodes):
            ax_main.annotate(str(i), (points[i, 0], points[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, zorder=6)
    
    # 绘制路径
    if len(tour) > 0:
        # 创建颜色映射用于路径段
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 10))
        route_segments = execution_history['route_segments']
        
        # 绘制路径段
        for i in range(len(tour)):
            current_node = tour[i]
            next_node = tour[(i + 1) % len(tour)]
            
            # 确定当前属于哪个路径段
            color_idx = 0
            for seg_idx, segment in enumerate(route_segments):
                if current_node in segment:
                    color_idx = seg_idx % len(colors)
                    break
            
            # 计算路径段样式
            start_point = points[current_node]
            end_point = points[next_node]
            
            # 开放路径返回depot用虚线
            line_style = '-' if not attribute_o or next_node != 0 else '--'
            line_width = 3 if next_node != 0 else 2
            
            # 检查该步骤是否有约束违反
            step_violations = execution_history['violations'][i + 1] if i + 1 < len(execution_history['violations']) else []
            has_violation = len(step_violations) > 0
            line_color = 'red' if has_violation else colors[color_idx]
            alpha = 0.9 if has_violation else 0.8
            
            ax_main.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                        color=line_color, linestyle=line_style, linewidth=line_width,
                        alpha=alpha, zorder=2)
            
            # 添加箭头指示方向
            mid_point = (start_point + end_point) / 2
            direction = end_point - start_point
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            ax_main.annotate('', xy=tuple(mid_point + direction * 0.02), xytext=tuple(mid_point - direction * 0.02),
                           arrowprops=dict(arrowstyle='->', color=line_color, lw=2), zorder=4)
            
            # 在路径上标记步骤编号
            ax_main.annotate(str(i + 1), tuple(mid_point), xytext=(0, 0), textcoords='offset points', 
                           fontsize=8, ha='center', va='center', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7), zorder=6)
    
    # 标记违反约束的节点
    violation_markers = {'capacity': ('x', 'red'), 'time_window': ('^', 'orange'), 'length': ('d', 'purple')}
    for i, step_violations in enumerate(execution_history['violations']):
        for violation_type, node_idx, *details in step_violations:
            if node_idx < len(points):
                point = points[node_idx]
                marker, color = violation_markers.get(violation_type, ('o', 'black'))
                ax_main.scatter(point[0], point[1], c=color, s=150, marker=marker, 
                               linewidths=3, zorder=7, label=f'{violation_type.replace("_", " ").title()} Violation')
    
    # 去除重复的图例标签
    handles, labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_main.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # 设置主图标题和标签
    title = f"{problem_type} Solution"
    if pred_cost is not None:
        title += f" (Cost: {pred_cost:.2f})"
    if gt_cost is not None:
        gap = ((pred_cost - gt_cost) / gt_cost * 100) if pred_cost and gt_cost else 0
        title += f" | GT: {gt_cost:.2f} | Gap: {gap:.1f}%"
    if title_suffix:
        title += f" | {title_suffix}"
    
    ax_main.set_title(title, fontsize=14, fontweight='bold')
    ax_main.set_xlabel('X Coordinate')
    ax_main.set_ylabel('Y Coordinate')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal', adjustable='box')
    
    # 显示约束信息和状态变化
    if show_constraints and points_with_features is not None and ax_info is not None:
        # 约束信息面板
        ax_info.axis('off')
        info_text = f"Problem Type: {problem_type}\n"
        info_text += f"Nodes: {num_nodes}\n"
        info_text += f"Routes: {len(execution_history['route_segments'])}\n"
        info_text += f"Steps: {len(execution_history['nodes']) - 1}\n\n"
        
        info_text += "Constraints:\n"
        if attribute_c:
            info_text += f"✓ Capacity (C)\n"
        if attribute_tw:
            info_text += f"✓ Time Windows (TW)\n"
        if attribute_o:
            info_text += f"✓ Open Routes (O)\n"
        if attribute_b:
            info_text += f"✓ Backhauls (B)\n"
        if attribute_l:
            info_text += f"✓ Length Limit (L)\n"
        
        info_text += f"\nViolations:\n"
        info_text += f"Capacity: {constraint_violations['capacity_violations']}\n"
        info_text += f"Time Window: {constraint_violations['time_window_violations']}\n"
        info_text += f"Length: {constraint_violations['length_violations']}\n"
        info_text += f"Total: {constraint_violations['total_violations']}\n"
        
        # 可行性判断
        is_feasible = constraint_violations['total_violations'] == 0
        feasibility_color = 'green' if is_feasible else 'red'
        feasibility_text = '✅ FEASIBLE' if is_feasible else '❌ INFEASIBLE'
        info_text += f"\nStatus: {feasibility_text}"
        
        # 最终状态信息
        final_load = execution_history['loads'][-1] if execution_history['loads'] else 0
        final_time = execution_history['times'][-1] if execution_history['times'] else 0
        total_distance = execution_history['distances'][-1] if execution_history['distances'] else 0
        info_text += f"\n\nFinal State:\n"
        info_text += f"Load: {final_load:.3f}\n"
        info_text += f"Time: {final_time:.3f}\n"
        info_text += f"Distance: {total_distance:.3f}"
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='lightgray', alpha=0.8))
        
        # 状态变化图表
        steps = range(len(execution_history['loads']))
        
        if ax_load and attribute_c:
            loads = execution_history['loads']
            ax_load.plot(steps, loads, 'b-', linewidth=2, marker='o', markersize=4)
            ax_load.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Empty')
            ax_load.set_title('Vehicle Load', fontsize=12)
            ax_load.set_ylabel('Remaining Capacity')
            ax_load.set_xlabel('Step')
            ax_load.grid(True, alpha=0.3)
            ax_load.legend()
            
            # 标记违反容量约束的点
            for i, step_violations in enumerate(execution_history['violations']):
                for violation_type, node_idx, *details in step_violations:
                    if violation_type == 'capacity' and i < len(loads):
                        ax_load.scatter(i, loads[i], c='red', s=50, marker='x', zorder=5)
        
        if ax_time and attribute_tw:
            times = execution_history['times']
            ax_time.plot(steps, times, 'g-', linewidth=2, marker='s', markersize=4)
            ax_time.set_title('Travel Time', fontsize=12)
            ax_time.set_ylabel('Cumulative Time')
            ax_time.set_xlabel('Step')
            ax_time.grid(True, alpha=0.3)
            
            # 标记违反时间窗约束的点
            for i, step_violations in enumerate(execution_history['violations']):
                for violation_type, node_idx, *details in step_violations:
                    if violation_type == 'time_window' and i < len(times):
                        ax_time.scatter(i, times[i], c='orange', s=50, marker='^', zorder=5)
        
        if ax_length and attribute_l:
            lengths = execution_history['lengths']
            ax_length.plot(steps, lengths, 'm-', linewidth=2, marker='^', markersize=4)
            ax_length.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Limit')
            ax_length.set_title('Remaining Length', fontsize=12)
            ax_length.set_ylabel('Length')
            ax_length.set_xlabel('Step')
            ax_length.grid(True, alpha=0.3)
            ax_length.legend()
            
            # 标记违反长度约束的点
            for i, step_violations in enumerate(execution_history['violations']):
                for violation_type, node_idx, *details in step_violations:
                    if violation_type == 'length' and i < len(lengths):
                        ax_length.scatter(i, lengths[i], c='purple', s=50, marker='d', zorder=5)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"VRP visualization saved to: {save_path}")
    
    return fig, execution_history


def greedy_solver_batch_pomo(adj_matrix_batch, points_with_features, temperature=1.0, problem_type="TSP", add_prior=False, 
                             test_mode=False, distance_matrices=None, current_epoch=None, max_epochs=None, debug=False):
    """
    POMO版本的批量TSP/VRP求解器 - 同时从多个起始点求解以获得更好的解
    
    POMO (Policy Optimization with Multiple Optima) 是一种增强型求解策略：
    - 对于TSP: 从所有节点作为起始点并行求解，获得 batch_size * num_nodes 个候选解
    - 对于VRP: 从所有客户节点作为起始点并行求解，获得 batch_size * (num_nodes-1) 个候选解
    - 支持温度控制的随机探索，平衡贪婪选择和随机探索
    - 完整支持VRP约束：容量(C)、时间窗(TW)、开放路径(O)、后装(B)、长度限制(L)
    
    Args:
        adj_matrix_batch: torch.Tensor, shape (batch_size, num_nodes, num_nodes)
            邻接矩阵批次，表示节点间的连接概率或权重
        points_with_features: torch.Tensor, shape (batch_size, num_nodes, feature_dim)
            节点特征矩阵：
            - TSP: shape (batch_size, num_nodes, 2) - [x, y] 坐标
            - VRP: shape (batch_size, num_nodes, 7) - [x, y, demand, early_tw, late_tw, route_open, length_limit]
        temperature: float, default=1.0
            探索温度参数：
            - 0.0: 完全贪婪选择
            - 1.0: 标准随机采样
            - >1.0: 更多随机探索
        problem_type: str, default="TSP"
            问题类型，用于确定约束类型：
            - "TSP": 旅行商问题
            - "CVRP": 有容量约束的车辆路径问题
            - "OVRP": 开放式车辆路径问题
            - "VRPTW": 带时间窗的车辆路径问题
            - 其他VRP变体组合
        add_prior: bool, default=False
            是否添加距离先验来增强邻接矩阵的连通性
        test_mode: bool, default=False
            是否为测试模式，影响对数概率的计算
        distance_matrices: torch.Tensor, shape (batch_size, num_nodes, num_nodes)
            距离矩阵，用于计算节点间实际距离
        current_epoch: int, default=None
            当前训练epoch，用于动态调整增强参数
        max_epochs: int, default=None
            最大训练epoch数，用于动态调整增强参数
        debug: bool, default=False
            是否启用调试模式，输出详细的诊断信息
    
    Returns:
        tours: torch.Tensor, shape (total_tours, max_tour_length)
            所有生成的路径，其中：
            - total_tours = batch_size * num_starts
            - num_starts = num_nodes (TSP) 或 num_nodes-1 (VRP)
            - max_tour_length 根据问题类型动态确定
        log_probs: torch.Tensor, shape (total_tours,)
            每个路径的累积对数概率，用于强化学习
    """
    # 第一步：预处理和参数设置
    
    # 调试信息：输入数据诊断
    if debug:
        print(f"\n=== POMO求解器调试信息 ===")
        print(f"问题类型: {problem_type}")
        print(f"温度参数: {temperature}")
        print(f"输入形状: adj_matrix_batch {adj_matrix_batch.shape}, points_with_features {points_with_features.shape}")
        print(f"adj_matrix_batch范围: min={adj_matrix_batch.min().item():.6f}, max={adj_matrix_batch.max().item():.6f}")
        print(f"adj_matrix_batch是否包含nan: {torch.isnan(adj_matrix_batch).any()}")
        print(f"adj_matrix_batch是否包含inf: {torch.isinf(adj_matrix_batch).any()}")
        
        # 检查邻接矩阵的概率分布
        adj_finite = adj_matrix_batch[torch.isfinite(adj_matrix_batch)]
        if len(adj_finite) > 0:
            print(f"有限值统计: mean={adj_finite.mean().item():.6f}, std={adj_finite.std().item():.6f}")
        print(f"零值比例: {(adj_matrix_batch == 0).float().mean().item():.3f}")
        print(f"接近1的比例: {(adj_matrix_batch > 0.9).float().mean().item():.3f}")
        print(f"================================\n")
    
    if add_prior:
        adj_matrix_batch = enhance_adjacency_matrix(adj_matrix_batch, distance_matrices=distance_matrices, connectivity_boost=0.5,
                                                   current_epoch=current_epoch, max_epochs=max_epochs)

    batch_size, num_nodes, _ = adj_matrix_batch.shape
    device = adj_matrix_batch.device
    
    # 解析问题类型，确定需要处理的约束类型
    attribute_c = 'C' in problem_type or problem_type in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW", "OVRPB", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]  # 容量约束
    attribute_tw = 'TW' in problem_type  # 时间窗约束
    attribute_o = 'O' in problem_type and problem_type.startswith('O')  # 开放路径约束
    attribute_b = 'B' in problem_type  # 后装约束
    attribute_l = 'L' in problem_type  # 长度限制约束
    
    # 第二步：POMO维度扩展 - 为每个起始点创建独立的求解实例
    if problem_type != "TSP" and num_nodes > 1:
        # VRP问题：只从客户节点开始（排除depot节点0）
        actual_starts = num_nodes - 1  # 客户节点数量
        # 扩展邻接矩阵: (batch_size, num_nodes, num_nodes) -> (batch_size * (num_nodes-1), num_nodes, num_nodes)
        expanded_adj = adj_matrix_batch.unsqueeze(1).expand(-1, actual_starts, -1, -1)
        expanded_adj = expanded_adj.reshape(batch_size * actual_starts, num_nodes, num_nodes)
    else:
        # TSP问题：从所有节点开始
        actual_starts = num_nodes
        expanded_adj = adj_matrix_batch.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        expanded_adj = expanded_adj.reshape(batch_size * num_nodes, num_nodes, num_nodes)
    
    total_tours = expanded_adj.shape[0]  # 总的求解实例数
    
    # 第三步：特征处理和验证
    feature_dim = points_with_features.shape[-1]

    expanded_features = points_with_features.unsqueeze(1).expand(-1, actual_starts, -1, -1)
    expanded_features = expanded_features.reshape(batch_size * actual_starts, num_nodes, feature_dim)
    
    if feature_dim == 2:
        # TSP场景：只有坐标信息
        if problem_type != "TSP":
            raise ValueError(f"VRP问题必须包含7维特征(坐标、需求、时间窗等)，当前只有{feature_dim}维。") 
        expanded_coords = expanded_features
        
        # TSP不需要VRP约束特征
        expanded_demands = None
        expanded_early_tw = None
        expanded_late_tw = None
        expanded_route_open = None
        expanded_length_limit = None
        
    elif feature_dim == 7:
        # VRP场景：完整的7维特征 [x, y, demand, early_tw, late_tw, route_open, length_limit] 
        # 分离各种特征
        expanded_coords = expanded_features[:, :, :2]  # 坐标
        expanded_demands = expanded_features[:, :, 2]  # 需求量
        expanded_early_tw = expanded_features[:, :, 3]  # 时间窗早期界限
        expanded_late_tw = expanded_features[:, :, 4]  # 时间窗晚期界限
        expanded_route_open = expanded_features[:, :, 5]  # 开放路径标志
        expanded_length_limit = expanded_features[:, :, 6]  # 路径长度限制
    else:
        raise ValueError(f"不支持的特征维度: {feature_dim}。期望2维（TSP）或7维（VRP）。")

    # 第四步：初始化求解状态
    # 创建起始节点索引
    if problem_type == "TSP":
        start_nodes = torch.arange(num_nodes, device=device).repeat(batch_size)
    else:
        # VRP：从客户节点1到num_nodes-1开始（排除depot节点0）
        start_nodes = torch.arange(1, num_nodes, device=device).repeat(batch_size, 1).flatten()
    
    # 确定路径长度上限
    if problem_type == "TSP":
        max_tour_length = num_nodes + 1  # TSP：访问所有节点+回到起点
    else:
        # VRP：考虑多次往返depot的最坏情况
        max_tour_length = max(num_nodes + 1, 2 * num_nodes + 10)
    
    # 初始化路径存储
    tours = torch.full((total_tours, max_tour_length), -1, dtype=torch.long, device=device)
    tours[:, 0] = start_nodes  # 设置起始节点
    tour_lengths = torch.ones(total_tours, dtype=torch.long, device=device)  # 当前路径长度
    current_nodes = start_nodes.clone()  # 当前位置
    log_probs = torch.zeros(total_tours, device=device) if not test_mode else None  # 累积对数概率
    
    # 第五步：初始化节点访问跟踪
    if problem_type == "TSP":
        # TSP：跟踪所有节点的访问状态
        visited_mask = torch.zeros(total_tours, num_nodes, dtype=torch.bool, device=device)
        # 标记起始节点为已访问
        visited_mask.scatter_(1, start_nodes.unsqueeze(1), True)
    else:
        # VRP：只跟踪客户节点的访问状态（depot可以多次访问）
        customer_visited_mask = torch.zeros(total_tours, num_nodes - 1, dtype=torch.bool, device=device)
        # 如果起始节点是客户节点，标记为已访问
        customer_start_indices = start_nodes - 1  # 转换为客户节点索引（0到num_nodes-2）
        customer_visited_mask.scatter_(1, customer_start_indices.unsqueeze(1), True)
    
    # 第六步：初始化VRP状态变量
    loads = torch.ones(total_tours, device=device)  # 车辆载重（1.0表示满载）
    times = torch.zeros(total_tours, device=device)  # 当前时间
    
    # 初始化路径长度约束
    if expanded_length_limit is not None and attribute_l:
        # 使用客户节点的长度限制作为初始值（避免depot的0值）
        initial_lengths = expanded_length_limit[:, 1]  # 使用第一个客户节点的长度限制
    else:
        initial_lengths = torch.full((total_tours,), 3.0, device=device)
    
    remaining_lengths = initial_lengths.clone()  # 剩余可用长度
    current_segment_distances = torch.zeros(total_tours, device=device)  # 当前路径段累积距离
    
    round_error_epsilon = 0.000001  # 数值比较容差
    
    # 第七步：主路径构建循环
    max_steps = max_tour_length - 2  # 预留空间给最终返回步骤
    
    # 添加路径完成状态跟踪
    finished_tours = torch.zeros(total_tours, dtype=torch.bool, device=device)
    
    for step in range(max_steps):
        # 检查路径构建完成条件
        if problem_type == "TSP":
            # TSP：固定步数（访问所有节点）
            if step >= num_nodes - 1:
                break
        else:
            # VRP：检查是否所有客户节点都已被访问
            all_customers_visited = customer_visited_mask.all(dim=1)  # shape: (total_tours,)
            
            # 更新完成状态：当前在depot且所有客户已访问
            at_depot = (current_nodes == 0)
            newly_finished = at_depot & all_customers_visited & ~finished_tours
            finished_tours = finished_tours | newly_finished
            
            if finished_tours.all():
                break  # 所有实例都已完成
        
        # 获取当前节点到所有节点的边权重
        batch_indices = torch.arange(total_tours, device=device)
        current_edges = expanded_adj[batch_indices, current_nodes]  # shape: (total_tours, num_nodes)
        
        # 第八步：构建节点选择掩码
        if problem_type == "TSP":
            # TSP：已访问的节点不能再次访问
            ninf_mask = torch.where(visited_mask, 
                                   torch.tensor(-float('inf'), device=device), 
                                   torch.zeros_like(visited_mask, dtype=torch.float))
        else:
            # VRP：构建更复杂的掩码
            ninf_mask = torch.zeros(total_tours, num_nodes, dtype=torch.float, device=device)
            
            # 屏蔽已访问的客户节点
            for i in range(num_nodes - 1):  # 遍历所有客户节点
                customer_idx = i  # 在customer_visited_mask中的索引
                node_idx = i + 1  # 在原图中的节点索引
                visited_customers = customer_visited_mask[:, customer_idx]
                ninf_mask[visited_customers, node_idx] = float('-inf')
            
            # depot处理：如果在depot且存在客户未访问，则禁止留在depot
            at_depot = (current_nodes == 0)
            all_customers_visited = customer_visited_mask.all(dim=1)
            depot_and_not_done = at_depot & ~all_customers_visited
            ninf_mask[depot_and_not_done, 0] = float('-inf')
        
        # 第九步：应用VRP约束
        if problem_type != "TSP":
            # 容量约束检查
            if attribute_c and expanded_demands is not None:
                # 检查剩余载重是否足够满足各节点需求
                demand_too_large = loads.unsqueeze(1) + round_error_epsilon < expanded_demands
                ninf_mask[demand_too_large] = float('-inf')
            
            # 时间窗约束检查
            if attribute_tw:
                # 计算到达各节点的时间
                current_points = expanded_coords[batch_indices, current_nodes]  # 当前位置坐标
                distances = torch.sqrt(torch.sum((current_points.unsqueeze(1) - expanded_coords) ** 2, dim=-1))
                arrival_times = times.unsqueeze(1) + distances
                
                # 屏蔽会违反时间窗晚期界限的节点
                time_violations = arrival_times > expanded_late_tw
                # 对于没有时间窗约束的节点（late_tw=0），不应用此约束
                no_tw_mask = expanded_late_tw == 0
                time_violations[no_tw_mask] = False
                ninf_mask[time_violations] = float('-inf')
            
            # 路径长度约束检查
            if attribute_l:
                current_points = expanded_coords[batch_indices, current_nodes]
                distances = torch.sqrt(torch.sum((current_points.unsqueeze(1) - expanded_coords) ** 2, dim=-1))  # (total_tours, num_nodes)
                
                if attribute_o:
                    # 开放路径：只需检查到目标节点的距离
                    length_violations = remaining_lengths.unsqueeze(1) - round_error_epsilon < distances  # (total_tours, num_nodes)
                else:
                    # 封闭路径：需要考虑返回depot的距离
                    depot_coords = expanded_coords[:, 0, :]  # depot坐标
                    return_distances = torch.sqrt(torch.sum((expanded_coords - depot_coords.unsqueeze(1)) ** 2, dim=-1))
                    
                    # 对于非depot节点，需要额外的返回距离
                    is_depot = torch.arange(num_nodes, device=device).unsqueeze(0).expand(total_tours, -1) == 0
                    total_distances = distances.clone()
                    total_distances[~is_depot] += return_distances[~is_depot]
                    
                    length_violations = remaining_lengths.unsqueeze(1) - round_error_epsilon < total_distances
                
                ninf_mask[length_violations] = float('-inf')
        
        # 开放路径特殊处理：确保depot始终可达以避免死锁
        if attribute_o:
            ninf_mask[:, 0] = 0.0  # depot永远可达
            # 但如果当前在depot且所有客户已访问，则这些路径已完成
            # 不需要额外处理，因为已通过finished_tours跟踪
        
        # 关键修复：对于已完成的路径，提供一个安全的选择（保持在depot）
        if problem_type != "TSP":
            # 对于已完成的路径，只允许选择depot（保持当前位置）
            ninf_mask[finished_tours, :] = float('-inf')  # 先屏蔽所有选择
            ninf_mask[finished_tours, 0] = 0.0  # 只允许选择depot
        
        # 第十步：节点选择 - 添加数值稳定性检查
        masked_edges = current_edges + ninf_mask
        
        # 检查输入数据的有效性（但-inf是正常的掩码，不需要修复）
        # 只检查current_edges是否有问题，ninf_mask中的-inf是正常的
        if torch.isnan(current_edges).any() or torch.isinf(current_edges).any():
            print(f"WARNING: current_edges包含nan或正inf（这不正常），将进行修复")
            print(f"current_edges范围: min={current_edges.min().item():.6f}, max={current_edges.max().item():.6f}")
            current_edges = torch.nan_to_num(current_edges, nan=0.0, posinf=1.0, neginf=0.0)
            masked_edges = current_edges + ninf_mask
        
        # 确保温度参数的有效性
        safe_temperature = max(temperature, 1e-6) if temperature > 0 else 1.0
        
        if temperature <= 0.0:
            # 贪婪选择
            # 检查是否所有选择都被屏蔽（这是真正的问题）
            all_masked = torch.all(masked_edges == float('-inf'), dim=-1)
            if all_masked.any():
                print(f"WARNING: {all_masked.sum().item()} 个样本的所有选择都被屏蔽，将提供安全选择")
                print(f"这通常表示VRP约束过于严格或求解器陷入死锁状态")
                # 为完全屏蔽的样本提供一个安全选择（通常是depot或第一个可用节点）
                masked_edges[all_masked, 0] = 0.0
            
            next_nodes = torch.argmax(masked_edges, dim=-1)
            edge_probs = F.softmax(masked_edges, dim=-1)
        else:
            # 基于温度的随机选择
            scaled_logits = masked_edges / safe_temperature
            
            # 限制logits的范围以防止数值溢出（但保留-inf掩码）
            # 只限制非-inf的值
            finite_mask = torch.isfinite(scaled_logits)
            if finite_mask.any():
                scaled_logits[finite_mask] = torch.clamp(scaled_logits[finite_mask], min=-50.0, max=50.0)
            
            # 检查是否所有选择都被屏蔽（这是真正的问题）
            all_masked = torch.all(scaled_logits == float('-inf'), dim=-1)
            if all_masked.any():
                print(f"WARNING: {all_masked.sum().item()} 个样本的所有选择都被屏蔽，将提供安全选择")
                print(f"这通常表示VRP约束过于严格或求解器陷入死锁状态")
                
                # 额外调试信息：分析为什么所有选择被屏蔽
                if debug:
                    print(f"=== 详细屏蔽分析 ===")
                    sample_idx = all_masked.nonzero()[0].item()
                    
                    # 添加并行训练相关的调试信息
                    print(f"并行训练环境信息:")
                    print(f"  设备: {device}")
                    print(f"  当前步骤: {step}")
                    print(f"  总实例数: {total_tours}")
                    print(f"  出现屏蔽的样本数: {all_masked.sum().item()}")
                    print(f"  屏蔽比例: {(all_masked.sum().item() / total_tours) * 100:.1f}%")
                    
                    # 检查分布式训练状态
                    if torch.distributed.is_initialized():
                        print(f"  分布式训练: rank={torch.distributed.get_rank()}/{torch.distributed.get_world_size()}")
                        print(f"  当前进程ID: {torch.distributed.get_rank()}")
                    else:
                        print(f"  非分布式训练")
                    
                    # 检查tensor的设备一致性
                    print(f"  tensor设备一致性:")
                    print(f"    ninf_mask: {ninf_mask.device}")
                    print(f"    current_nodes: {current_nodes.device}")
                    print(f"    loads: {loads.device}")
                    print(f"    expanded_coords: {expanded_coords.device}")
                    
                    # 检查内存使用情况
                    if device.type == 'cuda':
                        print(f"  GPU内存使用:")
                        print(f"    已分配: {torch.cuda.memory_allocated(device) / 1024**2:.1f}MB")
                        print(f"    缓存: {torch.cuda.memory_reserved(device) / 1024**2:.1f}MB")
                    
                    print(f"分析样本 {sample_idx}:")
                    print(f"current_node: {current_nodes[sample_idx].item()}")
                    print(f"ninf_mask状态: {(ninf_mask[sample_idx] == float('-inf')).sum().item()}/{ninf_mask.shape[1]} 个被屏蔽")
                    print(f"current_edges[{sample_idx}]: min={current_edges[sample_idx].min().item():.6f}, max={current_edges[sample_idx].max().item():.6f}")
                    
                    # 分析每个约束的贡献
                    sample_ninf = ninf_mask[sample_idx]  # shape: (num_nodes,)
                    
                    if problem_type != "TSP":
                        print(f"VRP状态:")
                        print(f"  当前载重: {loads[sample_idx].item():.3f}")
                        print(f"  当前时间: {times[sample_idx].item():.3f}")
                        print(f"  剩余长度: {remaining_lengths[sample_idx].item():.3f}")
                        print(f"  已访问客户: {customer_visited_mask[sample_idx].sum().item()}/{customer_visited_mask.shape[1]}")
                        print(f"  当前在depot: {(current_nodes[sample_idx] == 0).item()}")
                        print(f"  所有客户已访问: {customer_visited_mask[sample_idx].all().item()}")
                        
                        # 添加批次数据分布分析
                        print(f"\n批次数据分布分析:")
                        print(f"  所有样本的当前节点分布: {torch.bincount(current_nodes, minlength=num_nodes).tolist()}")
                        print(f"  所有样本的载重分布: min={loads.min().item():.3f}, max={loads.max().item():.3f}, mean={loads.mean().item():.3f}")
                        if attribute_tw:
                            print(f"  所有样本的时间分布: min={times.min().item():.3f}, max={times.max().item():.3f}, mean={times.mean().item():.3f}")
                        if attribute_l:
                            print(f"  所有样本的剩余长度分布: min={remaining_lengths.min().item():.3f}, max={remaining_lengths.max().item():.3f}, mean={remaining_lengths.mean().item():.3f}")
                        
                        # 检查同时屏蔽的样本是否有相似的状态
                        if all_masked.sum() > 1:
                            print(f"\n多样本屏蔽分析:")
                            masked_indices = all_masked.nonzero().flatten()
                            print(f"  被屏蔽的样本索引: {masked_indices.tolist()}")
                            
                            # 分析被屏蔽样本的状态相似性
                            masked_current_nodes = current_nodes[masked_indices]
                            masked_loads = loads[masked_indices]
                            
                            print(f"  被屏蔽样本的当前节点: {masked_current_nodes.tolist()}")
                            print(f"  被屏蔽样本的载重: {masked_loads.tolist()}")
                            
                            # 检查是否所有被屏蔽的样本都在相同状态
                            all_same_node = (masked_current_nodes == masked_current_nodes[0]).all()
                            all_same_load = torch.allclose(masked_loads, masked_loads[0], rtol=1e-5)
                            
                            print(f"  所有被屏蔽样本在相同节点: {all_same_node.item()}")
                            print(f"  所有被屏蔽样本载重相同: {all_same_load.item()}")
                            
                            if all_same_node and all_same_load:
                                print("  ⚠️  所有被屏蔽样本状态完全相同！可能是POMO扩展导致的重复状态")
                            else:
                                print("  ✓  被屏蔽样本状态不同，可能是独立的约束违反")
                        
                        # 分析每个约束的屏蔽情况
                        print(f"\n约束分析:")
                        
                        # 0. 约束贡献统计（整体分析）
                        print(f"  整体约束贡献统计:")
                        total_blocked = (sample_ninf == float('-inf')).sum().item()
                        
                        # 统计各约束的贡献
                        constraint_stats = {}
                        
                        # 1. 已访问客户节点分析
                        visited_mask_contribution = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                        for i in range(num_nodes - 1):  # 遍历所有客户节点
                            customer_idx = i  # 在customer_visited_mask中的索引
                            node_idx = i + 1  # 在原图中的节点索引
                            if customer_visited_mask[sample_idx, customer_idx]:
                                visited_mask_contribution[node_idx] = True
                        print(f"  已访问客户约束屏蔽: {visited_mask_contribution.sum().item()}/{num_nodes} 个节点 {visited_mask_contribution.nonzero().flatten().tolist()}")
                        constraint_stats['visited_customers'] = visited_mask_contribution.sum().item()
                        
                        # 2. Depot约束分析
                        at_depot = (current_nodes[sample_idx] == 0)
                        all_customers_visited = customer_visited_mask[sample_idx].all()
                        depot_constraint_active = at_depot and not all_customers_visited
                        print(f"  Depot约束: at_depot={at_depot.item()}, all_customers_visited={all_customers_visited.item()}, 屏蔽depot={depot_constraint_active}")
                        constraint_stats['depot_constraint'] = int(depot_constraint_active)
                        
                        # 3. 容量约束分析
                        capacity_blocked = 0
                        if attribute_c and expanded_demands is not None:
                            current_load = loads[sample_idx]
                            demands_sample = expanded_demands[sample_idx]  # shape: (num_nodes,)
                            capacity_violations = current_load + round_error_epsilon < demands_sample
                            capacity_blocked = capacity_violations.sum().item()
                            print(f"  容量约束屏蔽: {capacity_blocked}/{num_nodes} 个节点")
                            print(f"    当前载重: {current_load.item():.3f}")
                            print(f"    节点需求: {demands_sample.tolist()}")
                            print(f"    违反容量的节点: {capacity_violations.nonzero().flatten().tolist()}")
                        constraint_stats['capacity_constraint'] = capacity_blocked
                        
                        # 4. 时间窗约束分析
                        time_blocked = 0
                        if attribute_tw:
                            current_time = times[sample_idx]
                            current_point = expanded_coords[sample_idx, current_nodes[sample_idx]]
                            all_points = expanded_coords[sample_idx]  # shape: (num_nodes, 2)
                            distances = torch.sqrt(torch.sum((current_point.unsqueeze(0) - all_points) ** 2, dim=-1))
                            arrival_times = current_time + distances
                            late_tw_sample = expanded_late_tw[sample_idx]  # shape: (num_nodes,)
                            
                            time_violations = arrival_times > late_tw_sample
                            no_tw_mask = late_tw_sample == 0
                            time_violations[no_tw_mask] = False
                            time_blocked = time_violations.sum().item()
                            print(f"  时间窗约束屏蔽: {time_blocked}/{num_nodes} 个节点")
                            print(f"    当前时间: {current_time.item():.3f}")
                            print(f"    到达时间: {arrival_times.tolist()}")
                            print(f"    时间窗上限: {late_tw_sample.tolist()}")
                            print(f"    违反时间窗的节点: {time_violations.nonzero().flatten().tolist()}")
                        constraint_stats['time_window_constraint'] = time_blocked
                        
                        # 5. 路径长度约束分析
                        length_blocked = 0
                        if attribute_l:
                            current_remaining = remaining_lengths[sample_idx]
                            current_point = expanded_coords[sample_idx, current_nodes[sample_idx]]
                            all_points = expanded_coords[sample_idx]  # shape: (num_nodes, 2)
                            distances = torch.sqrt(torch.sum((current_point.unsqueeze(0) - all_points) ** 2, dim=-1))
                            
                            if attribute_o:
                                # 开放路径：只需检查到目标节点的距离
                                length_violations = current_remaining - round_error_epsilon < distances
                                length_blocked = length_violations.sum().item()
                                print(f"  路径长度约束屏蔽 (开放路径): {length_blocked}/{num_nodes} 个节点")
                                print(f"    剩余长度: {current_remaining.item():.3f}")
                                print(f"    到各节点距离: {distances.tolist()}")
                                print(f"    违反长度的节点: {length_violations.nonzero().flatten().tolist()}")
                            else:
                                # 封闭路径：需要考虑返回depot的距离
                                depot_coords = expanded_coords[sample_idx, 0]  # depot坐标
                                return_distances = torch.sqrt(torch.sum((all_points - depot_coords.unsqueeze(0)) ** 2, dim=-1))
                                
                                is_depot = torch.arange(num_nodes, device=device) == 0
                                total_distances = distances.clone()
                                total_distances[~is_depot] += return_distances[~is_depot]
                                
                                length_violations = current_remaining - round_error_epsilon < total_distances
                                length_blocked = length_violations.sum().item()
                                print(f"  路径长度约束屏蔽 (封闭路径): {length_blocked}/{num_nodes} 个节点")
                                print(f"    剩余长度: {current_remaining.item():.3f}")
                                print(f"    到各节点距离: {distances.tolist()}")
                                print(f"    返回depot距离: {return_distances.tolist()}")
                                print(f"    总距离需求: {total_distances.tolist()}")
                                print(f"    违反长度的节点: {length_violations.nonzero().flatten().tolist()}")
                        constraint_stats['length_constraint'] = length_blocked
                        
                        # 6. 完成状态约束分析
                        finished_blocked = 0
                        if finished_tours[sample_idx]:
                            finished_blocked = num_nodes - 1  # 除了depot外的所有节点
                            print(f"  完成状态约束: 路径已完成，只允许选择depot")
                        constraint_stats['finished_constraint'] = finished_blocked
                        
                        # 输出约束贡献统计
                        print(f"\n  约束贡献汇总:")
                        print(f"    总屏蔽节点数: {total_blocked}")
                        for constraint_name, blocked_count in constraint_stats.items():
                            if blocked_count > 0:
                                percentage = (blocked_count / num_nodes) * 100
                                print(f"    {constraint_name}: {blocked_count}/{num_nodes} ({percentage:.1f}%)")
                        
                        # 识别主要约束
                        max_constraint = max(constraint_stats.items(), key=lambda x: x[1])
                        if max_constraint[1] > 0:
                            print(f"    🔴 主要约束: {max_constraint[0]} (屏蔽了{max_constraint[1]}个节点)")
                        
                        # 检查是否存在约束重叠
                        total_individual_blocks = sum(constraint_stats.values())
                        if total_individual_blocks > total_blocked:
                            overlap = total_individual_blocks - total_blocked
                            print(f"    ⚠️  约束重叠: {overlap}个节点被多个约束同时屏蔽")
                        
                        # 7. 汇总分析
                        print(f"\n屏蔽汇总:")
                        for node_idx in range(num_nodes):
                            if sample_ninf[node_idx] == float('-inf'):
                                reasons = []
                                if problem_type != "TSP":
                                    # 检查各种约束
                                    if node_idx > 0 and customer_visited_mask[sample_idx, node_idx-1]:
                                        reasons.append("已访问")
                                    if node_idx == 0 and depot_constraint_active:
                                        reasons.append("depot约束")
                                    if attribute_c and expanded_demands is not None:
                                        if loads[sample_idx] + round_error_epsilon < expanded_demands[sample_idx, node_idx]:
                                            reasons.append("容量不足")
                                    if finished_tours[sample_idx] and node_idx != 0:
                                        reasons.append("路径已完成")
                                print(f"    节点{node_idx}: {' + '.join(reasons) if reasons else '未知原因'}")
                        
                        # 8. 建议诊断
                        print(f"\n诊断建议:")
                        if at_depot and not all_customers_visited and depot_constraint_active:
                            print("  - 在depot但还有客户未访问，却被depot约束阻止离开")
                            print("  - 建议检查: 是否所有客户都因其他约束而无法到达")
                        
                        if attribute_c and expanded_demands is not None:
                            remaining_customers = ~customer_visited_mask[sample_idx]
                            if remaining_customers.any():
                                remaining_demands = expanded_demands[sample_idx, 1:][remaining_customers]
                                if (remaining_demands > loads[sample_idx]).all():
                                    print("  - 所有未访问客户的需求都超过当前载重")
                                    print("  - 建议检查: 载重是否正确更新，或需求是否合理")
                        
                        if attribute_tw and times[sample_idx] > 0:
                            print("  - 可能因时间窗约束导致无法到达剩余节点")
                            print("  - 建议检查: 时间窗设置是否合理")
                        
                        if attribute_l and remaining_lengths[sample_idx] < 0.1:
                            print("  - 剩余路径长度过小")
                            print("  - 建议检查: 长度限制是否过于严格")
                        
                        # 9. 针对并行训练的特定建议
                        print(f"\n并行训练特定建议:")
                        if torch.distributed.is_initialized():
                            print("  - 分布式训练环境下，不同进程可能有不同的随机状态")
                            print("  - 建议检查: 各进程的随机种子是否正确设置")
                            print("  - 建议检查: 数据分布是否在不同进程间保持一致")
                        
                        # 根据主要约束类型提供针对性建议
                        if max_constraint[1] > 0:
                            main_constraint = max_constraint[0]
                            if main_constraint == 'capacity_constraint':
                                print("  - 主要问题: 容量约束")
                                print("    * 检查车辆容量设置是否合理")
                                print("    * 检查载重更新逻辑是否正确")
                                print("    * 考虑增加容量或减少需求")
                            elif main_constraint == 'time_window_constraint':
                                print("  - 主要问题: 时间窗约束")
                                print("    * 检查时间窗设置是否过于严格")
                                print("    * 检查时间更新逻辑是否正确")
                                print("    * 考虑放宽时间窗或调整速度")
                            elif main_constraint == 'length_constraint':
                                print("  - 主要问题: 路径长度约束")
                                print("    * 检查长度限制是否过于严格")
                                print("    * 检查长度更新逻辑是否正确")
                                print("    * 考虑增加长度限制或优化路径")
                            elif main_constraint == 'visited_customers':
                                print("  - 主要问题: 已访问客户约束")
                                print("    * 这通常是正常的，但如果所有客户都已访问还在继续，可能有问题")
                                print("    * 检查客户访问状态更新逻辑")
                            elif main_constraint == 'finished_constraint':
                                print("  - 主要问题: 路径完成约束")
                                print("    * 路径已完成但仍在寻找下一个节点")
                                print("    * 检查路径完成判断逻辑")
                        
                        # 10. 应急处理建议
                        print(f"\n应急处理建议:")
                        if all_masked.sum().item() > total_tours * 0.5:
                            print("  - 超过50%的样本被屏蔽，这是严重问题")
                            print("  - 建议: 临时禁用部分约束或调整约束参数")
                            print("  - 建议: 检查数据预处理是否正确")
                        
                        if torch.distributed.is_initialized() and all_masked.sum().item() > 0:
                            print("  - 在分布式环境中发现屏蔽问题")
                            print("  - 建议: 检查不同进程的数据是否同步")
                            print("  - 建议: 考虑使用单机测试验证问题")
                        
                        print("  - 如果问题持续存在，建议:")
                        print("    * 降低约束严格程度")
                        print("    * 增加数值容差 (round_error_epsilon)")
                        print("    * 检查数据集的合理性")
                        print("    * 使用更简单的问题类型进行测试")
                    
                    print(f"========================")
            
            # 为完全屏蔽的样本提供一个安全选择
            scaled_logits[all_masked, 0] = 0.0
        
            # 计算概率分布
            edge_probs = F.softmax(scaled_logits, dim=-1)
        
            # 现在检查softmax输出是否有问题（这里NaN才是真问题）
            if torch.isnan(edge_probs).any() or torch.isinf(edge_probs).any():
                print("ERROR: softmax后edge_probs包含nan或inf，这是真正的问题！")
                print(f"scaled_logits范围: min={scaled_logits[torch.isfinite(scaled_logits)].min().item() if torch.isfinite(scaled_logits).any() else 'all -inf'}")
                print(f"scaled_logits范围: max={scaled_logits[torch.isfinite(scaled_logits)].max().item() if torch.isfinite(scaled_logits).any() else 'all -inf'}")
                print(f"edge_probs包含nan: {torch.isnan(edge_probs).any()}")
                print(f"edge_probs包含inf: {torch.isinf(edge_probs).any()}")
                
                # 更详细的调试信息
                if debug:
                    print(f"=== Softmax问题详细分析 ===")
                    nan_samples = torch.isnan(edge_probs).any(dim=1).nonzero().flatten()
                    inf_samples = torch.isinf(edge_probs).any(dim=1).nonzero().flatten()
                    print(f"包含NaN的样本: {nan_samples.tolist()}")
                    print(f"包含Inf的样本: {inf_samples.tolist()}")
                    
                    if len(nan_samples) > 0:
                        sample_idx = nan_samples[0].item()
                        print(f"样本{sample_idx}的scaled_logits:")
                        logits_sample = scaled_logits[sample_idx]
                        finite_mask = torch.isfinite(logits_sample)
                        print(f"  有限值数量: {finite_mask.sum().item()}/{len(logits_sample)}")
                        if finite_mask.any():
                            print(f"  有限值范围: {logits_sample[finite_mask].min().item():.6f} - {logits_sample[finite_mask].max().item():.6f}")
                        print(f"  -inf数量: {(logits_sample == float('-inf')).sum().item()}")
                        print(f"  +inf数量: {(logits_sample == float('inf')).sum().item()}")
                        print(f"  nan数量: {torch.isnan(logits_sample).sum().item()}")
                    print(f"=============================")
                
                # 修复edge_probs
                edge_probs = torch.nan_to_num(edge_probs, nan=0.0, posinf=0.0, neginf=0.0)
                # 重新归一化
                edge_probs = edge_probs / (edge_probs.sum(dim=-1, keepdim=True) + 1e-8)
                print("已修复edge_probs")
            
            # 确保概率非负且归一化
            edge_probs = torch.clamp(edge_probs, min=1e-8)
            edge_probs = edge_probs / edge_probs.sum(dim=-1, keepdim=True)
            
            # 进行多项式采样，添加异常处理
            try:
                next_nodes = torch.multinomial(edge_probs, num_samples=1).squeeze(-1)
            except RuntimeError as e:
                print(f"ERROR: multinomial采样失败: {e}")
                print(f"edge_probs状态: min={edge_probs.min().item():.6f}, max={edge_probs.max().item():.6f}")
                print(f"edge_probs和: {edge_probs.sum(dim=-1).min().item():.6f} - {edge_probs.sum(dim=-1).max().item():.6f}")
                print(f"edge_probs是否包含nan: {torch.isnan(edge_probs).any()}")
                print(f"edge_probs是否包含inf: {torch.isinf(edge_probs).any()}")
                print(f"edge_probs是否包含负值: {(edge_probs < 0).any()}")
                # 回退到贪婪选择
                print("回退到贪婪选择")
                next_nodes = torch.argmax(scaled_logits, dim=-1)
        
            # 最终的安全检查：确保edge_probs有效（但这里应该很少触发了）
            if torch.isnan(edge_probs).any():
                print("ERROR: edge_probs仍然包含NaN，执行最终修复")
                edge_probs = torch.nan_to_num(edge_probs, nan=0.0, posinf=0.0, neginf=0.0)
                edge_probs = edge_probs / (edge_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            if not test_mode:
                # 计算并累积对数概率，添加更强的数值保护
                selected_probs = edge_probs.gather(1, next_nodes.unsqueeze(1)).squeeze(1)
                # 确保selected_probs为正数
                selected_probs = torch.clamp(selected_probs, min=1e-8)
                log_increment = torch.log(selected_probs)
                
                # 检查对数增量的有效性
                if torch.isnan(log_increment).any() or torch.isinf(log_increment).any():
                    print("WARNING: log_increment包含nan或inf，将设为安全值")
                    print(f"selected_probs范围: {selected_probs.min().item():.8f} - {selected_probs.max().item():.8f}")
                    log_increment = torch.nan_to_num(log_increment, nan=0.0, posinf=0.0, neginf=-10.0)
                
                log_probs += log_increment
        
        # 第十一步：更新路径和状态
        tours[batch_indices, tour_lengths] = next_nodes
        tour_lengths += 1
        
        # 更新节点访问状态
        if problem_type == "TSP":
            # TSP：标记新访问的节点
            visited_mask.scatter_(1, next_nodes.unsqueeze(1), True)
        else:
            # VRP：只更新客户节点的访问状态
            is_customer = next_nodes > 0  # 非depot节点
            if is_customer.any():
                customer_indices = (next_nodes - 1).clamp(0, num_nodes - 2)  # 转换为客户索引并防止越界
                # 修复：正确更新客户访问掩码
                customer_visited_mask[is_customer, customer_indices[is_customer]] = True
        
        # 更新VRP状态变量
        if problem_type != "TSP":
            at_depot_now = (next_nodes == 0)
            
            # 更新载重
            if attribute_c and expanded_demands is not None:
                consumed_demands = expanded_demands.gather(1, next_nodes.unsqueeze(1)).squeeze(1)
                loads -= consumed_demands
                loads[at_depot_now] = 1.0  # 在depot重新装载
            
            # 更新时间
            if attribute_tw and expanded_coords is not None and expanded_early_tw is not None:
                current_points = expanded_coords[batch_indices, current_nodes]
                next_points = expanded_coords[batch_indices, next_nodes]
                travel_times = torch.sqrt(torch.sum((next_points - current_points) ** 2, dim=-1))
                
                arrival_times = times + travel_times
                min_arrival_times = expanded_early_tw.gather(1, next_nodes.unsqueeze(1)).squeeze(1)
                times = torch.max(arrival_times, min_arrival_times)  # 等待时间窗开启
                times[at_depot_now] = 0.0  # 在depot重置时间
            
            # 更新路径长度
            if attribute_l and expanded_coords is not None:
                current_points = expanded_coords[batch_indices, current_nodes]
                next_points = expanded_coords[batch_indices, next_nodes]
                travel_distances = torch.sqrt(torch.sum((next_points - current_points) ** 2, dim=-1))
                
                remaining_lengths -= travel_distances
                current_segment_distances += travel_distances
                
                # 在depot时重置路径段状态
                if at_depot_now.any():
                    current_segment_distances[at_depot_now] = 0.0
                    remaining_lengths[at_depot_now] = initial_lengths[at_depot_now]
        
        # 更新当前节点位置
        current_nodes = next_nodes
        
        # 安全检查：避免无限循环
        if tour_lengths.max() >= max_tour_length - 1:
            break
    
    # 第十二步：路径收尾处理
    if problem_type == "TSP":
        # TSP：回到起始节点形成完整回路
        tours[batch_indices, tour_lengths] = start_nodes
        return_probs = expanded_adj[batch_indices, current_nodes, start_nodes]
        if not test_mode:
            log_probs += torch.log(return_probs + 1e-8)
        # 关键修复：更新tour_lengths以包含回到起点的节点
        tour_lengths += 1
    else:
        # VRP：如果不在depot，则返回depot
        not_at_depot = (current_nodes != 0)
        if not_at_depot.any():
            tours[not_at_depot, tour_lengths[not_at_depot]] = 0
            return_probs = expanded_adj[not_at_depot, current_nodes[not_at_depot], 0]
            if not test_mode:
                log_probs[not_at_depot] += torch.log(return_probs + 1e-8)
            tour_lengths[not_at_depot] += 1
    
    # 第十三步：路径格式化和输出
    # 移除填充的-1值，但保持使用-1作为无效位置的标识
    final_tours = []
    valid_lengths = []  # 记录每个路径的有效长度
    
    for i in range(total_tours):
        actual_length = int(tour_lengths[i].item())
        tour_i = tours[i, :actual_length]
        
        # 根据问题类型确定输出格式
        if problem_type == "TSP":
            min_length = num_nodes + 1
        else:
            min_length = max(num_nodes + 1, actual_length)
        
        # 关键修复：使用-1而不是0来填充，避免与有效节点编号混淆
        padded_tour = torch.full((min_length,), -1, dtype=torch.long, device=device)
        padded_tour[:len(tour_i)] = tour_i
        final_tours.append(padded_tour)
        valid_lengths.append(actual_length)
    
    # 统一所有路径的长度
    if final_tours:
        max_length = max(len(tour) for tour in final_tours)
        # 使用-1填充而不是0，保持一致性
        unified_tours = torch.full((total_tours, max_length), -1, dtype=torch.long, device=device)
        for i, tour in enumerate(final_tours):
            unified_tours[i, :len(tour)] = tour
        tours = unified_tours
    
    # 可选：返回有效长度信息（如果需要的话）
    # valid_lengths_tensor = torch.tensor(valid_lengths, dtype=torch.long, device=device)
    # return tours, log_probs, valid_lengths_tensor
    
    return tours, log_probs


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


def calculate_tour_cost_batch_pomo(tours, distance_matrices, problem_type="TSP"):
    """
    POMO版本的批量路径成本计算，支持开放路径
    Args:
        tours: torch.Tensor of shape (batch_size * num_nodes, num_nodes + 1) - 路径张量
        distance_matrices: torch.Tensor of shape (batch_size, num_nodes, num_nodes) - 距离矩阵
        problem_type: str - 问题类型，用于确定是否为开放路径
    Returns:
        costs: torch.Tensor of shape (batch_size * num_nodes,) - 每个路径的成本
    """
    batch_size, num_nodes, _ = distance_matrices.shape
    total_tours = tours.shape[0]  # batch_size * num_nodes
    device = tours.device
    
    # 确定是否为开放路径
    attribute_o = 'O' in problem_type and problem_type.startswith('O')
    
    # 扩展距离矩阵以匹配tours的维度
    # 从 (batch_size, num_nodes, num_nodes) 扩展为 (batch_size * num_nodes, num_nodes, num_nodes)
    if problem_type == "TSP":
        expanded_distances = distance_matrices.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        expanded_distances = expanded_distances.reshape(total_tours, num_nodes, num_nodes)
    else:
        # VRP问题：需要考虑实际的起始节点数
        actual_starts = num_nodes - 1 
        expanded_distances = distance_matrices.unsqueeze(1).expand(-1, actual_starts, -1, -1)
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
        
        # 对于开放路径（O属性），如果下一个节点是depot（通常是节点0），则距离设为0
        if attribute_o:
            # 根据VRPEnv.py中的逻辑：segment_lengths[self.selected_node_list.roll(dims=2, shifts=-1)==0] = 0
            # 这意味着返回depot的路径段成本为0
            is_return_to_depot = (next_nodes == 0)
            segment_costs[is_return_to_depot] = 0
        
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

def _extract_problem_type_from_path(file_path: str) -> str:
    """从文件路径中提取问题类型
    
    Args:
        file_path: 文件路径字符串
        
    Returns:
        str: 提取出的问题类型
    """
    file_path = file_path.upper()
    if "TSP" in file_path:
        return "TSP"
    elif "OVRPBLTW" in file_path:
        return "OVRPBLTW" 
    elif "VRPBLTW" in file_path:
        return "VRPBLTW"
    elif "OVRPBTW" in file_path:
        return "OVRPBTW"
    elif "OVRPLTW" in file_path:
        return "OVRPLTW"
    elif "VRPBTW" in file_path:
        return "VRPBTW"
    elif "VRPLTW" in file_path:
        return "VRPLTW"
    elif "OVRPBL" in file_path:
        return "OVRPBL"
    elif "VRPBL" in file_path:
        return "VRPBL"
    elif "OVRPB" in file_path:
        return "OVRPB"
    elif "VRPB" in file_path:
        return "VRPB"
    elif "OVRP" in file_path:
        return "OVRP"
    elif "CVRP" in file_path:
        return "CVRP"



class TSPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

    # ["TSP", "CVRP", "OVRP", "VRPB","VRPL", "VRPTW", "OVRPTW", "OVRPB", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW", "hybrid"]
    self.problem_type = self.args.problem_type   # 获取问题类型
    self.add_prior = self.args.add_prior

    # 处理混合任务
    if self.problem_type == "hybrid":
      # 混合任务：从配置文件中加载多个VRP变体
      hybrid_configs = getattr(self.args, 'hybrid_configs', [])

      # 检查数据文件是否存在，过滤掉不存在的文件
      valid_configs = []
      for config in hybrid_configs:
        if os.path.exists(config['data_file']):
          valid_configs.append(config)
        else:
          raise ValueError("没有找到有效的混合训练数据文件")  
      
      # 创建混合数据集
      self.train_dataset = HybridGraphDataset(
          data_configs=valid_configs,
          sparse_factor=self.args.sparse_factor,
          batch_size=self.args.batch_size
      )
      
      # 对于测试和验证集，使用第一个配置的数据集类型         
      self.problem_type_test = _extract_problem_type_from_path(self.args.test_split)
      problem_type_val = _extract_problem_type_from_path(self.args.validation_split)
      if problem_type_val != self.problem_type_test:
        raise ValueError("测试和验证集的问题类型不一致")
      primary_type = problem_type_val
      if primary_type == "TSP":
        self.test_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            sparse_factor=self.args.sparse_factor,
        )
        self.validation_dataset = TSPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
            sparse_factor=self.args.sparse_factor,
        )
      else:
        self.test_dataset = VRPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            sparse_factor=self.args.sparse_factor,
        )
        self.validation_dataset = VRPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
            sparse_factor=self.args.sparse_factor,
        )
      
      # 设置默认问题类型为第一个配置的类型（用于测试时）
      self.default_problem_type = primary_type
      
    elif self.problem_type != "TSP":
      self.train_dataset = VRPGraphDataset(
          data_file=os.path.join(self.args.storage_path, self.args.training_split),
          sparse_factor=self.args.sparse_factor
      )

      self.test_dataset = VRPGraphDataset(
          data_file=os.path.join(self.args.storage_path, self.args.test_split),
          sparse_factor=self.args.sparse_factor,
      )

      self.validation_dataset = VRPGraphDataset(
          data_file=os.path.join(self.args.storage_path, self.args.validation_split),
          sparse_factor=self.args.sparse_factor,
      )

    else:   # origin: TSP
      self.train_dataset = TSPGraphDataset(
          data_file=os.path.join(self.args.storage_path, self.args.training_split),
          sparse_factor=self.args.sparse_factor,
      )
 
      self.test_dataset = TSPGraphDataset(
          data_file=os.path.join(self.args.storage_path, self.args.test_split),
          sparse_factor=self.args.sparse_factor,
      )
      if self.args.test_debug:
        self.test_dataset.file_lines = self.test_dataset.file_lines[:1000]  # debug 使用

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
    
    # 新增的安全控制参数  
    self.rl_failure_count = 0  # 当前失败计数
    self.rl_skip_on_error = getattr(self.args, 'rl_skip_on_error', True)  # 遇到错误时是否跳过而不是崩溃
    
    # 数值稳定性参数
    self.max_logit_value = getattr(self.args, 'max_logit_value', 50.0)  # 最大logit值
    self.min_prob_value = getattr(self.args, 'min_prob_value', 1e-8)  # 最小概率值
    self.max_advantage = getattr(self.args, 'max_advantage', 20.0)  # 最大优势值
    
    # 调试参数
    self.rl_debug = getattr(self.args, 'rl_debug', False)  # 是否启用调试模式
    self.test_debug = getattr(self.args, 'test_debug', False)  # 是否启用测试调试模式
    
    print(f"强化学习配置: rl_loss_weight={self.rl_loss_weight}, temperature={self.pomo_temperature}")
    print(f"数值稳定性配置: max_logit={self.max_logit_value}, min_prob={self.min_prob_value}, max_advantage={self.max_advantage}")
    print(f"调试模式: rl_debug={self.rl_debug} {'(将输出详细诊断信息)' if self.rl_debug else ''}")
    print(f"测试调试: test_debug={self.test_debug} {'(测试时输出诊断信息)' if self.test_debug else ''}")

  def forward(self, x, adj, t, edge_index):
    return self.model(x, t, adj, edge_index)

  def get_collate_fn(self):
    """重写父类方法，返回自定义的collate函数"""
    if isinstance(self.train_dataset, HybridGraphDataset):
      # 混合数据集使用专门的collate函数
      return HybridCollateFunction()
    else:
      # 标准数据集使用通用的collate函数
      return self.custom_collate_fn

  def custom_collate_fn(self, batch):
    """
    自定义的collate函数，用于处理混合数据集的批次数据，支持变长数据的填充
    """
    try:
      # 混合数据集的批次包含问题类型信息
      if len(batch[0]) == 5:  # (sample_idx, points, adj_matrix, tour, problem_type)
        sample_indices, points, adj_matrices, tours, problem_types = zip(*batch)
        
        # 确保批次内的问题类型一致
        unique_types = set(problem_types)
        if len(unique_types) > 1:
          print(f"警告：批次中包含多种问题类型：{unique_types}，使用第一个类型")
        
        batch_problem_type = problem_types[0]
        
        # 处理变长数据的填充
        # 1. 找到最大的维度
        max_nodes = max(p.shape[0] for p in points)
        max_tour_length = max(t.shape[0] for t in tours)
        
        # 2. 填充points到相同大小
        padded_points = []
        for p in points:
          if p.shape[0] < max_nodes:
            # 用零填充
            pad_size = max_nodes - p.shape[0]
            padding = torch.zeros(pad_size, p.shape[1], dtype=p.dtype)
            padded_p = torch.cat([p, padding], dim=0)
          else:
            padded_p = p
          padded_points.append(padded_p)
        
        # 3. 填充adj_matrices到相同大小
        padded_adj_matrices = []
        for adj in adj_matrices:
          if adj.shape[0] < max_nodes:
            # 用零填充
            pad_size = max_nodes - adj.shape[0]
            padded_adj = torch.zeros(max_nodes, max_nodes, dtype=adj.dtype)
            padded_adj[:adj.shape[0], :adj.shape[1]] = adj
          else:
            padded_adj = adj
          padded_adj_matrices.append(padded_adj)
        
        # 4. 填充tours到相同长度
        padded_tours = []
        for t in tours:
          if t.shape[0] < max_tour_length:
            # 用-1填充（表示无效位置）
            pad_size = max_tour_length - t.shape[0]
            padding = torch.full((pad_size,), -1, dtype=t.dtype)
            padded_t = torch.cat([t, padding], dim=0)
          else:
            padded_t = t
          padded_tours.append(padded_t)
        
        # 5. 堆叠tensor
        sample_indices = torch.stack(sample_indices)
        points = torch.stack(padded_points)
        adj_matrices = torch.stack(padded_adj_matrices)
        tours = torch.stack(padded_tours)
        
        return sample_indices, points, adj_matrices, tours, batch_problem_type
      else:
        # 标准数据集的批次处理，也需要处理变长数据
        sample_indices, points, adj_matrices, tours = zip(*batch)
        
        # 处理变长数据的填充
        # 1. 找到最大的维度
        max_nodes = max(p.shape[0] for p in points)
        max_tour_length = max(t.shape[0] for t in tours)
        
        # 2. 填充points到相同大小
        padded_points = []
        for p in points:
          if p.shape[0] < max_nodes:
            # 用零填充
            pad_size = max_nodes - p.shape[0]
            padding = torch.zeros(pad_size, p.shape[1], dtype=p.dtype)
            padded_p = torch.cat([p, padding], dim=0)
          else:
            padded_p = p
          padded_points.append(padded_p)
        
        # 3. 填充adj_matrices到相同大小
        padded_adj_matrices = []
        for adj in adj_matrices:
          if adj.shape[0] < max_nodes:
            # 用零填充
            pad_size = max_nodes - adj.shape[0]
            padded_adj = torch.zeros(max_nodes, max_nodes, dtype=adj.dtype)
            padded_adj[:adj.shape[0], :adj.shape[1]] = adj
          else:
            padded_adj = adj
          padded_adj_matrices.append(padded_adj)
        
        # 4. 填充tours到相同长度
        padded_tours = []
        for t in tours:
          if t.shape[0] < max_tour_length:
            # 用-1填充（表示无效位置）
            pad_size = max_tour_length - t.shape[0]
            padding = torch.full((pad_size,), -1, dtype=t.dtype)
            padded_t = torch.cat([t, padding], dim=0)
          else:
            padded_t = t
          padded_tours.append(padded_t)
        
        # 5. 堆叠tensor
        sample_indices = torch.stack(sample_indices)
        points = torch.stack(padded_points)
        adj_matrices = torch.stack(padded_adj_matrices)
        tours = torch.stack(padded_tours)
        
        return sample_indices, points, adj_matrices, tours
    
    except Exception as e:
      print(f"ERROR in custom_collate_fn: {e}")
      print(f"Batch info - size: {len(batch)}")
      if len(batch) > 0:
        print(f"First item info - type: {type(batch[0])}, length: {len(batch[0])}")
        if hasattr(batch[0], '__iter__'):
          for i, item in enumerate(batch[0]):
            if hasattr(item, 'shape'):
              print(f"  Item {i}: shape {item.shape}, dtype {item.dtype}")
            else:
              print(f"  Item {i}: type {type(item)}, value {item}")
      raise

  def train_dataloader(self):
    batch_size = self.args.batch_size
    
    # 使用标准的PyTorch DataLoader以确保collate_fn参数生效
    import torch.utils.data
    
    if isinstance(self.train_dataset, HybridGraphDataset):
      # 混合数据集：使用自定义的BatchSampler确保批次内问题类型一致
      print("使用HybridBatchSampler确保批次内问题类型一致")
      
      batch_sampler = HybridBatchSampler(
          sampler=self.train_dataset,  # 传入数据集作为sampler
          batch_size=batch_size, 
          drop_last=True
      )
      # 设置shuffle
      batch_sampler.set_shuffle(True)
      
      train_dataloader = torch.utils.data.DataLoader(
          self.train_dataset,
          batch_sampler=batch_sampler,  # 使用自定义批次采样器
          num_workers=self.args.num_workers, 
          pin_memory=True,
          persistent_workers=True,
          collate_fn=HybridCollateFunction()
      )
    else:
      # 标准数据集：使用常规方式
      train_dataloader = torch.utils.data.DataLoader(
          self.train_dataset, 
          batch_size=batch_size, 
          shuffle=True,
          num_workers=self.args.num_workers, 
          pin_memory=True,
          persistent_workers=True, 
          drop_last=True,
          collate_fn=self.custom_collate_fn
      )
    
    return train_dataloader

  def test_dataloader(self):
    """重写test_dataloader以使用支持变长数据的collate函数"""
    batch_size = 1
    print("Test dataset size:", len(self.test_dataset))
    import torch.utils.data
    
    if isinstance(self.test_dataset, HybridGraphDataset):
      # 混合数据集：使用自定义的BatchSampler（但batch_size=1）
      print("测试阶段使用HybridBatchSampler")
      
      # 为测试创建一个batch_size=1的sampler
      batch_sampler = HybridBatchSampler(
          sampler=self.test_dataset,  # 传入数据集作为sampler
          batch_size=batch_size, 
          drop_last=False
      )
      # 测试阶段不需要打乱
      batch_sampler.set_shuffle(False)
      
      test_dataloader = torch.utils.data.DataLoader(
          self.test_dataset,
          batch_sampler=batch_sampler,
          collate_fn=HybridCollateFunction()
      )
    else:
      # 标准数据集：使用常规方式
      test_dataloader = torch.utils.data.DataLoader(
          self.test_dataset, 
          batch_size=batch_size, 
          shuffle=False,
          collate_fn=self.custom_collate_fn
      )
    return test_dataloader

  def val_dataloader(self):
    """重写val_dataloader以使用支持变长数据的collate函数"""
    batch_size = 1
    import torch.utils.data
    val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.args.validation_examples))
    print("Validation dataset size:", len(val_dataset))
    
    if isinstance(self.validation_dataset, HybridGraphDataset):
      # 混合数据集：使用自定义的BatchSampler（但batch_size=1）
      print("验证阶段使用HybridBatchSampler")
      
      # 注意：对于Subset，我们需要特殊处理
      # 创建一个简化的sampler，因为Subset不支持HybridBatchSampler
      val_dataloader = torch.utils.data.DataLoader(
          val_dataset, 
          batch_size=batch_size, 
          shuffle=False,
          collate_fn=HybridCollateFunction()  # 仍然使用混合collate函数
      )
    else:
      # 标准数据集：使用常规方式
      val_dataloader = torch.utils.data.DataLoader(
          val_dataset, 
          batch_size=batch_size, 
          shuffle=False,
          collate_fn=self.custom_collate_fn
      )
    return val_dataloader

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    
    # 处理混合数据集的批次数据
    if self.problem_type == "hybrid":
      # 混合数据集返回5个元素：sample_idx, points, adj_matrix, tour, problem_type
      if len(batch) == 5:
        _, points, adj_matrix, gt_tour, batch_problem_type = batch
        # 验证批次内问题类型的一致性
        if isinstance(batch_problem_type, (list, tuple)):
            unique_types = set(batch_problem_type)
            if len(unique_types) > 1:
                raise RuntimeError(f"严重错误: 批次内包含多种问题类型 {unique_types}。这表明批次采样器存在问题,请检查数据加载逻辑。")
            current_problem_type = batch_problem_type[0]
        else:
            current_problem_type = batch_problem_type
        
        # 记录批次问题类型成功信息
        if batch_idx % 100 == 0:  # 每100个批次记录一次
          print(f"Batch {batch_idx}: 问题类型 = {current_problem_type}, 批次大小 = {points.shape[0]}")
      else:
        # 兼容性处理：如果批次格式不匹配，使用默认问题类型
        _, points, adj_matrix, gt_tour = batch
        current_problem_type = getattr(self, 'default_problem_type', 'CVRP')
        print(f"警告：批次格式不匹配，使用默认问题类型 {current_problem_type}")
    else:
      # 非混合数据集的标准处理
      _, points, adj_matrix, gt_tour = batch
      current_problem_type = self.problem_type
    
    # 记录当前批次的问题类型（使用hash以便在tensorboard中显示）
    problem_type_hash = hash(current_problem_type) % 1000
    self.log("train/problem_type", problem_type_hash, on_step=True, on_epoch=False)
    # self.log("train/problem_type_name", current_problem_type, on_step=False, on_epoch=True)  # 注释掉，因为不能记录字符串
    
    # 验证数据形状的一致性
    batch_size = points.shape[0]
    if adj_matrix.shape[0] != batch_size or gt_tour.shape[0] != batch_size:
      raise ValueError(f"批次数据形状不一致：points {points.shape}, adj_matrix {adj_matrix.shape}, gt_tour {gt_tour.shape}")
    
    t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)

    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    if self.sparse:
      adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)
    xt = self.diffusion.sample(adj_matrix_onehot, t)
    # 确保xt是tensor类型（修复类型检查警告）
    if isinstance(xt, tuple):
        xt = xt[1]  # 如果返回tuple，取第二个元素
    xt = xt * 2 - 1  # 使用浮点数字面量避免类型错误
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
    
    # 更新条件检查，包含新的安全控制参数
    compute_rl = (self.rl_loss_weight > 0 and 
                  not self.sparse and 
                  batch_idx % self.rl_compute_frequency == 0)
    
    if compute_rl:  # 暂时只支持非稀疏图
        try:
            # 获取预测的邻接矩阵概率 - 添加数值稳定性检查
            # 首先检查x0_pred的数值范围
            if torch.isnan(x0_pred).any() or torch.isinf(x0_pred).any():
                raise ValueError("x0_pred包含nan或inf")
            
            # 限制x0_pred的范围以防止softmax溢出
            x0_pred_clamped = torch.clamp(x0_pred, min=-self.max_logit_value, max=self.max_logit_value)
            
            # 计算softmax概率
            x0_pred_prob = x0_pred_clamped.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            
            # 检查softmax输出的有效性
            if torch.isnan(x0_pred_prob).any() or torch.isinf(x0_pred_prob).any():
                raise ValueError("x0_pred_prob包含nan或inf")
            
            # 取边存在的概率 (第二个类别)
            adj_prob_matrix = x0_pred_prob[:, :, :, 1]  # shape: (batch_size, num_nodes, num_nodes)
            
            # 确保概率矩阵在有效范围内
            adj_prob_matrix = torch.clamp(adj_prob_matrix, min=self.min_prob_value, max=1.0 - self.min_prob_value)
            
            # 计算真实距离矩阵
            distance_matrices = calculate_euclidean_distance_batch(points[:, :, :2])
            
            # 检查距离矩阵的有效性
            if torch.isnan(distance_matrices).any() or torch.isinf(distance_matrices).any():
                raise ValueError("distance_matrices包含nan或inf")
            
            # 使用POMO版本的求解器获取路径和对数概率，支持探索
            pred_tours, log_probs = greedy_solver_batch_pomo(adj_prob_matrix, temperature=self.pomo_temperature, 
                points_with_features=points, problem_type=current_problem_type,
                add_prior=self.add_prior, distance_matrices=distance_matrices, test_mode=False,
                current_epoch=self.current_epoch, max_epochs=self.trainer.max_epochs if self.trainer else None,
                debug=self.rl_debug)  # 使用实例属性
            
            # 检查求解器输出的有效性
            if log_probs is None or torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                raise ValueError("求解器返回的log_probs无效")
            
            # 计算预测路径的成本
            pred_costs = calculate_tour_cost_batch_pomo(pred_tours, distance_matrices, problem_type=current_problem_type)   
            
            # 检查成本计算的有效性
            if torch.isnan(pred_costs).any() or torch.isinf(pred_costs).any():
                raise ValueError("pred_costs包含nan或inf")
            
            # 计算真实最优路径的成本作为基准
            gt_tours_list = []
            for b in range(points.shape[0]):
                gt_tour_b = gt_tour[b].cpu().numpy().tolist()
                gt_tour_b.append(gt_tour_b[0])  # 添加回到起点
                gt_tours_list.append(gt_tour_b)
            
            gt_costs = calculate_tour_cost_batch(gt_tours_list, distance_matrices)
            
            # 检查真实成本的有效性
            if torch.isnan(gt_costs).any() or torch.isinf(gt_costs).any() or (gt_costs <= 0).any():
                raise ValueError("gt_costs包含无效值")
            
            # 将gt_costs扩展以匹配POMO的维度
            if current_problem_type == "TSP":
                num_starts = points.shape[1] 
            else:
                num_starts = points.shape[1] - 1
            gt_costs_expanded = gt_costs.unsqueeze(1).expand(-1, num_starts).reshape(-1)
            
            # 计算奖励 (负的相对成本差异) - 添加数值保护
            cost_diff = pred_costs - gt_costs_expanded
            relative_cost_diff = cost_diff / (gt_costs_expanded + self.min_prob_value)
            
            # 限制相对成本差异的范围以防止极端值
            relative_cost_diff = torch.clamp(relative_cost_diff, min=-10.0, max=10.0)
            rewards = -relative_cost_diff  # 成本越低，奖励越高
            
            # 检查奖励的有效性
            if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                raise ValueError("rewards包含nan或inf")
            
            # 对于POMO，我们选择每个样本中最好的路径来计算基线
            rewards_reshaped = rewards.reshape(points.shape[0], num_starts)  # (batch_size, num_starts)
            best_rewards = torch.max(rewards_reshaped, dim=1)[0]  # (batch_size,)
            
            # 更新基线 (使用指数移动平均) - 添加数值保护
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
            
            # REINFORCE损失 (负的对数概率乘以优势) - 最终数值保护
            rl_loss_raw = -(log_probs * advantages.detach()).mean()
            
            if torch.isnan(rl_loss_raw) or torch.isinf(rl_loss_raw):
                raise ValueError("rl_loss_raw无效")
            
            rl_loss = rl_loss_raw
            
            # 重置失败计数（成功计算）
            self.rl_failure_count = 0
            
            # 记录强化学习相关指标
            self.log("train/rl_loss", rl_loss)
            self.log("train/avg_reward", rewards.mean())
            self.log("train/best_reward", best_rewards.mean())
            self.log("train/baseline", self.rl_baseline)
            self.log("train/avg_pred_cost", pred_costs.mean())
            self.log("train/best_pred_cost", pred_costs.reshape(points.shape[0], num_starts).min(dim=1)[0].mean())
            self.log("train/avg_gt_cost", gt_costs.mean())
            self.log("train/cost_gap_percent", (relative_cost_diff * 100).mean())
            self.log("train/best_cost_gap_percent", ((pred_costs.reshape(points.shape[0], num_starts).min(dim=1)[0] - gt_costs) / (gt_costs + 1e-8) * 100).mean())
            self.log("train/pomo_temperature", self.pomo_temperature)
            self.log("train/rl_failure_count", float(self.rl_failure_count))
            
        except Exception as e:
            # 增加失败计数
            self.rl_failure_count += 1
            error_msg = f"强化学习损失计算失败 ({self.rl_failure_count}): {e}"
            
            if self.rl_skip_on_error:
                print(f"WARNING: {error_msg}")
                rl_loss = torch.tensor(0.0, device=points.device)
                
                # 记录失败信息
                self.log("train/rl_failure_count", float(self.rl_failure_count))
            else:
                # 不跳过错误，重新抛出异常
                raise RuntimeError(error_msg)
        
        # 移除之前的重复代码块，因为我们现在统一在try-except中处理
    
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
    self.log("train/avg_diffusion_timestep", avg_t, on_step=True, on_epoch=True, sync_dist=True)
    
    # 记录动态connectivity_boost值
    if self.add_prior and self.current_epoch is not None and self.trainer and self.trainer.max_epochs:
        # 计算当前的动态boost值（使用与enhance_adjacency_matrix相同的逻辑）
        epoch_ratio = self.current_epoch / self.trainer.max_epochs
        decay_rate = 3.0
        initial_boost = 0.1  # 默认初始值
        dynamic_boost = initial_boost * torch.exp(torch.tensor(-decay_rate * epoch_ratio))
        min_boost = initial_boost * 0.01
        dynamic_boost = max(dynamic_boost.item(), min_boost)
        
        self.log("train/connectivity_boost", dynamic_boost, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/boost_decay_ratio", epoch_ratio, on_step=True, on_epoch=True, sync_dist=True)
    
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
    
    # 智能检测当前批次的问题类型
    if hasattr(self, 'problem_type_test'):
      current_problem_type = self.problem_type_test
    else:
      current_problem_type = self.problem_type
    
    # 处理混合数据集的批次数据（类似categorical_training_step）
    if current_problem_type == "hybrid" or len(batch) == 5:
      # 混合数据集返回5个元素：sample_idx, points, adj_matrix, tour, problem_type
      if len(batch) == 5:
        real_batch_idx, points, adj_matrix, gt_tour, batch_problem_type = batch
        # 对于混合测试，使用批次中的问题类型
        if isinstance(batch_problem_type, (list, tuple)):
          current_problem_type = batch_problem_type[0]
        else:
          current_problem_type = batch_problem_type
      else:
        # 兼容性处理：如果批次格式不匹配，使用默认问题类型
        real_batch_idx, points, adj_matrix, gt_tour = batch
        print(f"警告：测试批次格式不匹配，使用默认问题类型 {current_problem_type}")
    else:
      # 非混合数据集的标准处理
      real_batch_idx, points, adj_matrix, gt_tour = batch
    
    # 记录当前批次的问题类型
    self.log(f"{split}/problem_type", hash(current_problem_type) % 1000, on_step=True, on_epoch=False)
    
    if not self.sparse:
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
    else:
      # 对于稀疏图的处理，需要重新解析batch
      if self.problem_type == "hybrid" and len(batch) == 6:
        # 稀疏图的混合数据集可能有6个元素
        real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour, batch_problem_type = batch
        current_problem_type = batch_problem_type[0] if isinstance(batch_problem_type, (list, tuple)) else batch_problem_type
      else:
        # 标准稀疏图处理
        real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
        # current_problem_type已经在上面设置了
        
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
      adj_prob_matrix = torch.from_numpy(adj_mat).float().to(device)
    
      # 确保是单个样本的形状 (1, num_nodes, num_nodes)
      if adj_prob_matrix.dim() == 2:
        adj_prob_matrix = adj_prob_matrix.unsqueeze(0)
    
      # 使用POMO方法生成多个候选路径
    #   use_pomo = getattr(self.args, 'use_pomo', False)  # 测试时默认使用POMO
      test_temperature = getattr(self.args, 'test_temperature', 0)  # 测试时使用较小的温度以减少随机性 
    
      # 使用POMO版本生成路径
      points_tensor = torch.from_numpy(np_points).float().unsqueeze(0).to(device)  # (1, num_nodes, 2或7)
    
      # 检查并处理特征维度
      if points_tensor.shape[-1] == 2:
          # 只有2维坐标
          if current_problem_type != "TSP":  
              raise ValueError(f"VRP问题必须包含7维特征(坐标、需求、时间窗等)，当前只有{points_tensor.shape[-1]}维。" + 
                            "请检查数据集是否正确包含了所有必要的VRP约束特征。")
          else: 
              points_with_features = points_tensor
      elif points_tensor.shape[-1] == 7:
          # 已经有7维特征（VRP数据集），直接使用
          points_with_features = points_tensor
      else:
          # 其他维度，报错
          raise ValueError(f"不支持的特征维度: {points_tensor.shape[-1]}。期望2维（TSP）或7维（VRP）。")
    
      distance_matrices = calculate_euclidean_distance_batch(points_with_features[:, :, :2]) # 计算所有路径的成本
      pred_tours, _ = greedy_solver_batch_pomo(adj_prob_matrix, temperature=test_temperature, 
                                                points_with_features=points_with_features, 
                                                problem_type=current_problem_type,
                                                add_prior=self.add_prior,
                                                test_mode=True,
                                                distance_matrices=distance_matrices,
                                                current_epoch=None, max_epochs=None,  # 测试阶段不需要epoch信息
                                                debug=self.test_debug  # 新增测试调试开关
                                                )
      # pred_tours: (1 * num_nodes, num_nodes + 1) for TSP or (1 * (num_nodes-1), num_nodes + 1) for VRP
      pred_costs = calculate_tour_cost_batch_pomo(pred_tours, distance_matrices, problem_type=current_problem_type)
    
      # 选择成本最小的路径
      best_idx = torch.argmin(pred_costs)
      best_tour = pred_tours[best_idx].cpu().numpy()
    
      # 转换为列表格式，保留完整路径（包括回到起点的节点），过滤掉填充值
      best_tour_valid = best_tour[best_tour >= 0]  # 过滤掉-1的填充值
      solved_tours = [best_tour_valid.tolist()]
      
      # 转换为numpy数组
      solved_tours = np.array(solved_tours, dtype='int64')
    
      stacked_tours.append(solved_tours)

    solved_tours = np.concatenate(stacked_tours, axis=0)

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)

    total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
    all_solved_costs = [float(tsp_solver.evaluate(solved_tours[i])) for i in range(total_sampling)]
    best_solved_cost = min(all_solved_costs)

    # 可视化对比真实路径和预测路径
    debug_visualize_route_comparison(
        points_with_features=points_with_features,
        np_points=np_points,
        np_gt_tour=np_gt_tour,
        solved_tours=solved_tours,
        gt_cost=gt_cost,
        best_solved_cost=best_solved_cost,
        split=split,
        batch_idx=batch_idx,
        problem_type=current_problem_type,
        trainer=self.trainer,
        logger=self.logger,
        args=self.args,
        enable_debug=self.args.draw_route_comparison  # 设置为True以启用调试模式
    )

    # 计算额外的性能指标
    gap_percentage = ((best_solved_cost - gt_cost) / gt_cost) * 100
    all_gaps = [((cost - gt_cost) / gt_cost) * 100 for cost in all_solved_costs]
    avg_gap = np.mean(np.array(all_gaps))
    std_gap = np.std(np.array(all_gaps))
    
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
        f"{split}/best_solved_cost": best_solved_cost,
        f"{split}/diffusion_steps": self.args.inference_diffusion_steps,
        f"{split}/problem_type_name": current_problem_type,  # 记录问题类型
    }
    
    # 记录所有指标到TensorBoard和PyTorch Lightning
    for k, v in metrics.items():
        # 跳过best_solved_cost，因为我们会单独记录它以添加进度条显示
        if not k.endswith('/best_solved_cost'):
            # 跳过problem_type_name，因为它是字符串类型
            if not k.endswith('/problem_type_name'):
                self.log(k, v, on_epoch=True, sync_dist=True)
    
    # 特别标记最重要的指标用于进度条显示
    self.log(f"{split}/best_solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
    
    # 将结果保存到父类的test_outputs列表中（兼容PyTorch Lightning 2.0+）
    self.test_outputs.append(metrics)
    
    return metrics

  def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split):
    if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    
    # 优先使用checkpoint路径的父目录（如果有的话），否则使用当前logger的目录
    if hasattr(self.trainer, 'ckpt_path') and self.trainer.ckpt_path is not None:
        # 从checkpoint路径推断出实验目录
        ckpt_path = self.trainer.ckpt_path
        # 获取checkpoint目录的父目录（即版本目录）
        exp_save_dir = os.path.dirname(os.path.dirname(ckpt_path))
    else:
        # 如果没有checkpoint路径，使用当前logger的目录
        exp_save_dir = self.logger.log_dir if self.logger and hasattr(self.logger, 'log_dir') and self.logger.log_dir else './logs'
        
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')

  def on_train_start(self):
    """训练开始时的回调，显示数据集和批次统计信息"""
    super().on_train_start()
    
    if self.problem_type == "hybrid" and isinstance(self.train_dataset, HybridGraphDataset):
      print("\n" + "="*60)
      print("混合数据集训练统计信息")
      print("="*60)
      
      # 显示数据集配置
      print(f"总样本数: {len(self.train_dataset)}")
      print(f"总批次数: {len(self.train_dataset.batch_mappings)}")
      
      # 统计每种问题类型的批次数量
      type_counts = {}
      for problem_type in self.train_dataset.batch_types:
        type_counts[problem_type] = type_counts.get(problem_type, 0) + 1
      
      print(f"批次分布:")
      for problem_type, count in type_counts.items():
        percentage = (count / len(self.train_dataset.batch_types)) * 100
        print(f"  {problem_type}: {count} 批次 ({percentage:.1f}%)")
      
      print(f"批次大小: {self.args.batch_size}")
      print("✅ 确保每个批次内问题类型一致")
      print("="*60 + "\n")
    else:
      print(f"\n使用标准数据集训练: {self.problem_type}")
      print(f"总样本数: {len(self.train_dataset)}")
      print(f"批次大小: {self.args.batch_size}\n")


def enhance_adjacency_matrix(adj_matrix, min_prob=0.01, connectivity_boost=0.9, distance_matrices=None, current_epoch=None, max_epochs=None):
    """
    增强邻接矩阵的连通性，主要通过实际几何距离先验来改善低质量热力图
    
    当扩散模型训练初期输出的热力图质量较差时，这个函数可以：
    1. 确保所有边都有最小概率，避免完全断连
    2. 基于实际几何距离增强近邻连接的概率（距离先验）
    3. 提高求解器找到完整路径的成功率
    4. 随着训练进行，逐步减少对启发式信息的依赖
    
    Args:
        adj_matrix: torch.Tensor of shape (batch_size, num_nodes, num_nodes) - 原始邻接概率矩阵
        min_prob: float - 所有边的最小概率值，防止完全为0的边
        connectivity_boost: float - 基于距离的连通性增强系数（初始值）
        distance_matrices: torch.Tensor of shape (batch_size, num_nodes, num_nodes) - 距离矩阵，用于计算节点间实际距离
        current_epoch: int - 当前训练epoch（可选）
        max_epochs: int - 最大训练epoch数（可选）
    
    Returns:
        enhanced_matrix: torch.Tensor - 增强后的邻接矩阵
    """
    enhanced = adj_matrix.clone()
    
    # 步骤1: 确保最小概率，避免求解器遇到完全不可达的节点
    enhanced = torch.clamp(enhanced, min=min_prob)
    
    # 步骤2: 动态调整connectivity_boost参数
    dynamic_boost = connectivity_boost
    if current_epoch is not None and max_epochs is not None and max_epochs > 0:
        # 使用指数衰减策略：boost = initial_boost * exp(-decay_rate * epoch_ratio)
        # 这样可以在初期保持较高值，然后逐步减小
        epoch_ratio = current_epoch / max_epochs
        decay_rate = 2.0  # 衰减率，可以调整这个值来控制衰减速度
        dynamic_boost = connectivity_boost * torch.exp(torch.tensor(-decay_rate * epoch_ratio))
        
        # 设置最小值，避免完全消失
        min_boost = connectivity_boost * 0.01  # 保留初始值的1%作为最小值
        dynamic_boost = max(dynamic_boost.item(), min_boost)
        
        # 添加调试信息（可选）
        # if current_epoch % 10 == 0:  # 每10个epoch输出一次
        #     print(f"[Epoch {current_epoch}/{max_epochs}] Connectivity boost: {connectivity_boost:.4f} → {dynamic_boost:.4f} (ratio: {epoch_ratio:.3f})")
        
        # 可选：使用线性衰减策略（注释掉指数衰减，取消注释这部分来使用）
        # dynamic_boost = connectivity_boost * (1 - epoch_ratio)
        
        # 可选：使用余弦退火策略（注释掉指数衰减，取消注释这部分来使用）
        # import math
        # dynamic_boost = connectivity_boost * (1 + math.cos(math.pi * epoch_ratio)) / 2
    
    # 步骤3: 基于实际几何距离的先验增强
    if distance_matrices is not None:
        batch_size, num_nodes, _ = enhanced.shape
        device = enhanced.device
        
        # 确保distance_matrices在同一设备上
        if distance_matrices.device != device:
            distance_matrices = distance_matrices.to(device)
        
        # 计算距离权重：距离越小，权重越大
        # 使用负指数函数将距离转换为权重: exp(-distance/scale)
        # 这样可以确保近距离节点获得更高的权重
        
        # 首先计算距离的统计信息以确定合适的缩放因子
        # 排除对角线元素（自己到自己的距离为0）
        mask = torch.eye(num_nodes, device=device).bool().unsqueeze(0).expand(batch_size, -1, -1)
        masked_distances = distance_matrices.masked_select(~mask)
        
        # 使用距离的中位数或平均值作为缩放因子
        distance_scale = masked_distances.median().item()
        if distance_scale == 0:
            distance_scale = 1.0  # 防止除零
        
        # 计算基于距离的权重
        # 使用负指数函数：exp(-distance/scale)
        distance_weights = torch.exp(-distance_matrices / distance_scale)
        
        # 将对角线元素设为0（节点到自己的增强权重为0）
        diagonal_mask = torch.eye(num_nodes, device=device).bool().unsqueeze(0).expand(batch_size, -1, -1)
        distance_weights[diagonal_mask] = 0
        
        # 应用动态调整后的距离权重增强
        enhanced = (1-dynamic_boost) * enhanced + dynamic_boost * distance_weights
        
    else:
        # 如果没有提供距离矩阵，回退到原来的索引距离方式
        # 但这种情况应该避免，因为索引距离的假设通常不准确
        print("警告：没有提供距离矩阵，回退到索引距离启发式")
        batch_size, num_nodes, _ = enhanced.shape
        
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        # 计算节点索引距离的倒数作为先验权重
                        # 索引距离越小，增强越多（假设索引相近的节点空间距离也相近）
                        index_distance_factor = 1.0 / (abs(i - j) + 1)
                        enhanced[b, i, j] += dynamic_boost * index_distance_factor
    
    # 步骤4: 归一化到合理范围，避免概率过大
    enhanced = torch.clamp(enhanced, min=0, max=1)
    
    return enhanced


def calculate_tour_cost_batch_single(tours, distance_matrices, problem_type="TSP"):
    """
    非POMO版本的批量路径成本计算，支持开放路径，每个样本只有一个路径
    Args:
        tours: torch.Tensor of shape (batch_size, max_tour_length) - 路径张量
        distance_matrices: torch.Tensor of shape (batch_size, num_nodes, num_nodes) - 距离矩阵
        problem_type: str - 问题类型，用于确定是否为开放路径
    Returns:
        costs: torch.Tensor of shape (batch_size,) - 每个路径的成本
    """
    batch_size, _ = tours.shape
    device = tours.device
    
    # 确定是否为开放路径
    attribute_o = 'O' in problem_type and problem_type.startswith('O')
    
    # 计算路径成本
    costs = torch.zeros(batch_size, device=device)
    
    # 使用张量操作计算所有路径段的成本
    for i in range(tours.shape[1] - 1):  # max_tour_length - 1 个路径段
        current_nodes = tours[:, i]      # shape: (batch_size,)
        next_nodes = tours[:, i + 1]     # shape: (batch_size,)
        
        # 跳过填充的部分（值为-1或0的无效段）
        valid_segments = (current_nodes >= 0) & (next_nodes >= 0) & (current_nodes != next_nodes)
        
        if valid_segments.any():
            # 获取对应的距离
            batch_indices = torch.arange(batch_size, device=device)
            segment_costs = distance_matrices[batch_indices, current_nodes, next_nodes]
            
            # 对于开放路径（O属性），如果下一个节点是depot（通常是节点0），则距离设为0
            if attribute_o:
                is_return_to_depot = (next_nodes == 0)
                segment_costs[is_return_to_depot] = 0
            
            # 只累加有效路径段的成本
            costs[valid_segments] += segment_costs[valid_segments]
    
    return costs


def debug_visualize_route_comparison(
    points_with_features,
    np_points,
    np_gt_tour,
    solved_tours,
    gt_cost,
    best_solved_cost,
    split,
    batch_idx,
    problem_type,
    trainer=None,
    logger=None,
    args=None,
    enable_debug=False
):
    """
    调试模式的路径可视化对比函数
    
    Args:
        points_with_features: 包含特征的点数据
        np_points: 点的坐标数据，shape: (num_nodes, 2)
        np_gt_tour: 真实路径，shape: (num_nodes,)
        solved_tours: 预测路径列表
        gt_cost: 真实路径成本
        best_solved_cost: 预测路径成本
        split: 数据集分割（train/val/test）
        batch_idx: 批次索引
        problem_type: 问题类型
        trainer: 训练器对象
        logger: 日志记录器
        args: 参数对象
        enable_debug: 是否启用调试模式
    """
    if not enable_debug:
        return
    
    # 直接使用已经从batch中获取的特征数据
    # points_with_features 已经在前面从 points_tensor 获得，包含了正确的特征维度
    current_features = points_with_features.cpu().numpy()[0] if points_with_features is not None else None
    
    # 获取路径数据
    points = np_points[:, :2]  # shape: (num_nodes, 2) 
    gt_path = np_gt_tour  # shape: (num_nodes,)
    pred_path = solved_tours[0]  # shape: (num_nodes,)
    
    # 确定保存路径
    if trainer is not None and hasattr(trainer, 'ckpt_path') and trainer.ckpt_path is not None:
        ckpt_path = trainer.ckpt_path
        exp_save_dir = os.path.dirname(os.path.dirname(ckpt_path))
    else:
        exp_save_dir = logger.log_dir if logger and hasattr(logger, 'log_dir') and logger.log_dir else './logs'
        
    vis_path = os.path.join(exp_save_dir, 'route_visualization')
    os.makedirs(vis_path, exist_ok=True)
    
    # 构建文件名信息
    model_info = f"v{getattr(logger, 'version', 'unknown')}_{getattr(logger, 'name', 'model')}"
    use_pomo = getattr(args, 'use_pomo', True) if args else True
    pomo_info = "pomo" if use_pomo else "greedy"
    test_temp = getattr(args, 'test_temperature', 0.0) if args else 0.0
    
    # 可视化预测路径
    pred_filename = f'pred_route_{model_info}_{pomo_info}_temp{test_temp}_{split}_batch{batch_idx}.png'
    pred_save_path = os.path.join(vis_path, pred_filename)
    
    title_suffix = f"{pomo_info.upper()}, temp={test_temp}"
    
    try:
        fig_pred, execution_history = visualize_vrp_solution(
            points=points,
            tour=pred_path,
            points_with_features=current_features,
            problem_type=problem_type,
            gt_cost=gt_cost,
            pred_cost=best_solved_cost,
            save_path=pred_save_path,
            title_suffix=title_suffix,
            show_constraints=True,
            figsize=(15, 10)
        )
        plt.close(fig_pred)  # 关闭图形以释放内存
        
        # 从execution_history中获取约束违反信息
        violations = execution_history['constraint_violations']
        
        # 记录约束违反信息
        if violations['total_violations'] > 0:
            print(f"⚠️  Solution has {violations['total_violations']} constraint violations:")
            if violations['capacity_violations'] > 0:
                print(f"   - Capacity violations: {violations['capacity_violations']}")
            if violations['time_window_violations'] > 0:
                print(f"   - Time window violations: {violations['time_window_violations']}")
            if violations['length_violations'] > 0:
                print(f"   - Length violations: {violations['length_violations']}")
            
            # 记录详细的违反信息
            print(f"📊 Execution Summary:")
            print(f"   - Total steps: {len(execution_history['nodes']) - 1}")
            print(f"   - Route segments: {len(execution_history['route_segments'])}")
            print(f"   - Final load: {execution_history['loads'][-1] if execution_history['loads'] else 0:.3f}")
            print(f"   - Final time: {execution_history['times'][-1] if execution_history['times'] else 0:.3f}")
            print(f"   - Total distance: {execution_history['distances'][-1] if execution_history['distances'] else 0:.3f}")
        else:
            print("✅ Solution is feasible (no constraint violations)")
            print(f"📊 Execution Summary:")
            print(f"   - Total steps: {len(execution_history['nodes']) - 1}")
            print(f"   - Route segments: {len(execution_history['route_segments'])}")
            print(f"   - Final load: {execution_history['loads'][-1] if execution_history['loads'] else 0:.3f}")
            print(f"   - Final time: {execution_history['times'][-1] if execution_history['times'] else 0:.3f}")
            print(f"   - Total distance: {execution_history['distances'][-1] if execution_history['distances'] else 0:.3f}")
        
        # 如果有真实路径，也可视化对比
        if gt_path is not None and len(gt_path) > 0:
            gt_filename = f'gt_route_{model_info}_{split}_batch{batch_idx}.png'
            gt_save_path = os.path.join(vis_path, gt_filename)
            
            fig_gt, _ = visualize_vrp_solution(
                points=points,
                tour=gt_path,
                points_with_features=current_features,
                problem_type=problem_type,
                pred_cost=gt_cost,
                save_path=gt_save_path,
                title_suffix="Ground Truth",
                show_constraints=True,
                figsize=(15, 10)
            )
            plt.close(fig_gt)
            
            # 创建对比图
            comparison_filename = f'comparison_{model_info}_{pomo_info}_temp{test_temp}_{split}_batch{batch_idx}.png'
            comparison_save_path = os.path.join(vis_path, comparison_filename)
            
            fig_comp, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # 左侧：真实路径
            ax1 = axes[0]
            ax1.scatter(points[:, 0], points[:, 1], c='lightblue', s=100, marker='o', 
                       edgecolors='blue', linewidths=1, zorder=3)
            if problem_type != "TSP":
                ax1.scatter(points[0, 0], points[0, 1], c='red', s=200, marker='s', 
                           label='Depot', edgecolors='black', linewidths=2, zorder=5)
            
            for i in range(len(gt_path)):
                start = points[gt_path[i]]
                end = points[gt_path[(i + 1) % len(gt_path)]]
                ax1.plot([start[0], end[0]], [start[1], end[1]], 'r-', linewidth=2)
            
            ax1.set_title(f'Ground Truth (Cost: {gt_cost:.2f})', fontsize=14, fontweight='bold')
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal', adjustable='box')
            
            # 右侧：预测路径
            ax2 = axes[1]
            ax2.scatter(points[:, 0], points[:, 1], c='lightblue', s=100, marker='o', 
                       edgecolors='blue', linewidths=1, zorder=3)
            if problem_type != "TSP":
                ax2.scatter(points[0, 0], points[0, 1], c='red', s=200, marker='s', 
                           label='Depot', edgecolors='black', linewidths=2, zorder=5)
            
            for i in range(len(pred_path)):
                start = points[pred_path[i]]
                end = points[pred_path[(i + 1) % len(pred_path)]]
                ax2.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2)
            
            gap = ((best_solved_cost - gt_cost) / gt_cost * 100) if gt_cost > 0 else 0
            feasible_status = "feasible" if violations['total_violations'] == 0 else f"infeasible({violations['total_violations']})"
            ax2.set_title(f'Prediction (Cost: {best_solved_cost:.2f}, Gap: {gap:.1f}%) {feasible_status}', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            plt.savefig(comparison_save_path, dpi=300, bbox_inches='tight')
            plt.close(fig_comp)
            
            print(f"Comparison visualization saved to: {comparison_save_path}")
            
    except Exception as e:
        print(f"VRP可视化失败: {e}")
        # 回退到简单可视化
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
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
        
        fallback_filename = f'fallback_route_comparison_{model_info}_{pomo_info}_temp{test_temp}_{split}_batch{batch_idx}.png'
        plt.savefig(os.path.join(vis_path, fallback_filename))
        plt.close(fig)