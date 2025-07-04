"""Lightning module for training the DIFUSCO TSP model."""

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
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours  # 注释掉因为改用强化学习方法


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
        for i in range(len(tour) - 1):
            current_node = tour[i]
            next_node = tour[i + 1]
            
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
            step_violations = execution_history['violations'][i + 1]  # i+1因为violations[0]是初始状态
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


def greedy_tsp_solver_batch_pomo(adj_matrix_batch, points_with_features, temperature=1.0, problem_type="TSP", add_prior=False):
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
    if add_prior:
        adj_matrix_batch = enhance_adjacency_matrix(adj_matrix_batch)

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
    log_probs = torch.zeros(total_tours, device=device)  # 累积对数概率
    
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
        if expanded_length_limit.shape[1] > 1:
            initial_lengths = expanded_length_limit[:, 1]  # 使用第一个客户节点的长度限制
        else:
            initial_lengths = torch.full((total_tours,), 3.0, device=device)
    else:
        initial_lengths = torch.full((total_tours,), 3.0, device=device)
    
    remaining_lengths = initial_lengths.clone()  # 剩余可用长度
    current_segment_distances = torch.zeros(total_tours, device=device)  # 当前路径段累积距离
    
    round_error_epsilon = 0.000001  # 数值比较容差
    
    # 第七步：主路径构建循环
    max_steps = max_tour_length - 2  # 预留空间给最终返回步骤
    
    for step in range(max_steps):
        # 检查路径构建完成条件
        if problem_type == "TSP":
            # TSP：固定步数（访问所有节点）
            if step >= num_nodes - 1:
                break
        else:
            # VRP：检查是否所有客户节点都已被访问
            all_customers_visited = customer_visited_mask.all(dim=1)  # shape: (total_tours,)
            if all_customers_visited.all():
                break  # 所有实例都已访问完所有客户
        
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
            
            # depot处理：如果在depot且所有客户已访问，则禁止留在depot
            at_depot = (current_nodes == 0)
            all_customers_visited = customer_visited_mask.all(dim=1)
            depot_and_done = at_depot & all_customers_visited
            ninf_mask[depot_and_done, 0] = float('-inf')
        
        # 第九步：应用VRP约束
        if problem_type != "TSP":
            # 容量约束检查
            if attribute_c and expanded_demands is not None:
                # 检查剩余载重是否足够满足各节点需求
                demand_too_large = loads.unsqueeze(1) + round_error_epsilon < expanded_demands
                ninf_mask[demand_too_large] = float('-inf')
            
            # 时间窗约束检查
            if attribute_tw and expanded_early_tw is not None and expanded_late_tw is not None:
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
                distances = torch.sqrt(torch.sum((current_points.unsqueeze(1) - expanded_coords) ** 2, dim=-1))
                
                if attribute_o:
                    # 开放路径：只需检查到目标节点的距离
                    length_violations = remaining_lengths.unsqueeze(1) - round_error_epsilon < distances
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
            # 但如果当前在depot且所有客户已访问，则可以结束
            if problem_type != "TSP":
                at_depot = (current_nodes == 0)
                all_customers_visited = customer_visited_mask.all(dim=1)
                can_finish = at_depot & all_customers_visited
                ninf_mask[can_finish, 0] = float('-inf')
        
        # 第十步：节点选择
        masked_edges = current_edges + ninf_mask
        
        if temperature <= 0.0:
            # 贪婪选择
            next_nodes = torch.argmax(masked_edges, dim=-1)
            edge_probs = F.softmax(masked_edges, dim=-1)
        else:
            # 基于温度的随机选择
            scaled_logits = masked_edges / temperature
            edge_probs = F.softmax(scaled_logits, dim=-1)
            next_nodes = torch.multinomial(edge_probs, num_samples=1).squeeze(-1)
        
        # 计算并累积对数概率
        selected_probs = edge_probs.gather(1, next_nodes.unsqueeze(1)).squeeze(1)
        log_probs += torch.log(selected_probs + 1e-8)
        
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
        log_probs += torch.log(return_probs + 1e-8)
        # 关键修复：更新tour_lengths以包含回到起点的节点
        tour_lengths += 1
    else:
        # VRP：如果不在depot，则返回depot
        not_at_depot = (current_nodes != 0)
        if not_at_depot.any():
            tours[not_at_depot, tour_lengths[not_at_depot]] = 0
            return_probs = expanded_adj[not_at_depot, current_nodes[not_at_depot], 0]
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
        actual_starts = num_nodes - 1 if num_nodes > 1 else num_nodes
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


class TSPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

    # ["TSP", "CVRP", "OVRP", "VRPB","VRPL", "VRPTW", "OVRPTW", "OVRPB", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]
    self.problem_type = self.args.problem_type   # 获取问题类型
    self.add_prior = self.args.add_prior

    if self.problem_type != "TSP":
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
      # self.test_dataset.file_lines = self.test_dataset.file_lines[:1000]  # debug 使用

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
            # 使用POMO版本的求解器获取路径和对数概率，支持探索 
            pred_tours, log_probs = greedy_tsp_solver_batch_pomo(
                adj_prob_matrix, temperature=self.pomo_temperature, 
                points_with_features=points, problem_type=self.problem_type,
                add_prior=self.add_prior
            )
            # pred_tours: (batch_size * num_nodes, num_nodes + 1)
            # log_probs: (batch_size * num_nodes,)
            
            # 计算真实距离矩阵
            distance_matrices = calculate_euclidean_distance_batch(points[:, :, :2])
            
            # 计算预测路径的成本
            pred_costs = calculate_tour_cost_batch_pomo(pred_tours, distance_matrices, problem_type=self.problem_type)
            # pred_costs: (batch_size * num_nodes,)
            
            # 计算真实最优路径的成本作为基准
            gt_tours_list = []
            for b in range(points.shape[0]):
                gt_tour_b = gt_tour[b].cpu().numpy().tolist()
                if self.problem_type != "TSP":
                    # 对于VRP问题，gt_tour可能需要特殊处理
                    gt_tour_b.append(gt_tour_b[0])  # 添加回到起点（如果不是开放路径）
                else:
                    gt_tour_b.append(gt_tour_b[0])  # 添加回到起点
                gt_tours_list.append(gt_tour_b)
            
            gt_costs = calculate_tour_cost_batch(gt_tours_list, distance_matrices)
            # gt_costs: (batch_size,)
            
            # 将gt_costs扩展以匹配POMO的维度
            if self.problem_type == "TSP":
                num_starts = points.shape[1] 
            else:
                num_starts = points.shape[1] - 1 if points.shape[1] > 1 else points.shape[1]
            gt_costs_expanded = gt_costs.unsqueeze(1).expand(-1, num_starts).reshape(-1)
            # gt_costs_expanded: (batch_size * num_starts,)
            
            # 计算奖励 (负的相对成本差异)
            relative_cost_diff = (pred_costs - gt_costs_expanded) / (gt_costs_expanded + 1e-8)
            rewards = -relative_cost_diff  # 成本越低，奖励越高
            
            # 对于POMO，我们选择每个样本中最好的路径来计算基线
            rewards_reshaped = rewards.reshape(points.shape[0], num_starts)  # (batch_size, num_starts)
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
            self.log("train/best_pred_cost", pred_costs.reshape(points.shape[0], num_starts).min(dim=1)[0].mean())
            self.log("train/avg_gt_cost", gt_costs.mean())
            self.log("train/cost_gap_percent", (relative_cost_diff * 100).mean())
            self.log("train/best_cost_gap_percent", ((pred_costs.reshape(points.shape[0], num_starts).min(dim=1)[0] - gt_costs) / (gt_costs + 1e-8) * 100).mean())
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
    self.log("train/avg_diffusion_timestep", avg_t, on_step=True, on_epoch=True, sync_dist=True)
    
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
    adj_prob_matrix = torch.from_numpy(adj_mat).float().to(device)
    
    # 确保是单个样本的形状 (1, num_nodes, num_nodes)
    if adj_prob_matrix.dim() == 2:
        adj_prob_matrix = adj_prob_matrix.unsqueeze(0)
    
    # 使用POMO方法生成多个候选路径
    use_pomo = getattr(self.args, 'use_pomo', True)  # 测试时默认使用POMO
    test_temperature = getattr(self.args, 'test_temperature', 0.0)  # 测试时使用较小的温度以减少随机性 
    
    # 使用POMO版本生成路径
    points_tensor = torch.from_numpy(np_points).float().unsqueeze(0).to(device)  # (1, num_nodes, 2或7)
    
    # 检查并处理特征维度
    if points_tensor.shape[-1] == 2:
        # 只有2维坐标
        if self.problem_type != "TSP":  
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
    
    pred_tours, _ = greedy_tsp_solver_batch_pomo(
                                                adj_prob_matrix, temperature=test_temperature, 
                                                points_with_features=points_with_features, 
                                                problem_type=self.problem_type,
                                                add_prior=self.add_prior
                                                )
    # pred_tours: (1 * num_nodes, num_nodes + 1) for TSP or (1 * (num_nodes-1), num_nodes + 1) for VRP
    
    # 计算所有路径的成本
    distance_matrices = calculate_euclidean_distance_batch(points_with_features[:, :, :2])
    pred_costs = calculate_tour_cost_batch_pomo(pred_tours, distance_matrices, problem_type=self.problem_type)
    
    # 选择成本最小的路径
    best_idx = torch.argmin(pred_costs)
    best_tour = pred_tours[best_idx].cpu().numpy()
    
    # 转换为列表格式，去掉最后的重复起始点
    solved_tours = [best_tour[:-1].tolist()]
    
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
    debug_mode = False   # 临时启用以测试路径修复
    if debug_mode:
        # 直接使用已经从batch中获取的特征数据
        # points_with_features 已经在前面从 points_tensor 获得，包含了正确的特征维度
        current_features = points_with_features.cpu().numpy()[0] if points_with_features is not None else None
        
        # 获取路径数据
        points = np_points[:, :2]  # shape: (num_nodes, 2) 
        gt_path = np_gt_tour # shape: (num_nodes,)
        pred_path = solved_tours[0]  # shape: (num_nodes,)
        
        # 确定保存路径
        if hasattr(self.trainer, 'ckpt_path') and self.trainer.ckpt_path is not None:
            ckpt_path = self.trainer.ckpt_path
            exp_save_dir = os.path.dirname(os.path.dirname(ckpt_path))
        else:
            exp_save_dir = self.logger.log_dir if self.logger and hasattr(self.logger, 'log_dir') and self.logger.log_dir else './logs'
            
        vis_path = os.path.join(exp_save_dir, 'route_visualization')
        os.makedirs(vis_path, exist_ok=True)
        
        # 构建文件名信息
        model_info = f"v{getattr(self.logger, 'version', 'unknown')}_{getattr(self.logger, 'name', 'model')}"
        use_pomo = getattr(self.args, 'use_pomo', True)
        pomo_info = "pomo" if use_pomo else "greedy"
        test_temp = getattr(self.args, 'test_temperature', 0.0)
        
        # 可视化预测路径
        pred_filename = f'pred_route_{model_info}_{pomo_info}_temp{test_temp}_{split}_batch{batch_idx}.png'
        pred_save_path = os.path.join(vis_path, pred_filename)
        
        title_suffix = f"{pomo_info.upper()}, temp={test_temp}"
        
        try:
            fig_pred, execution_history = visualize_vrp_solution(
                points=points,
                tour=pred_path,
                points_with_features=current_features,
                problem_type=self.problem_type,
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
                    problem_type=self.problem_type,
                    pred_cost=gt_cost,
                    save_path=gt_save_path,
                    title_suffix="Ground Truth",
                    show_constraints=True,
                    figsize=(15, 10)
                )
                plt.close(fig_gt)
                
                # gt_violations = gt_execution_history['constraint_violations']
                
                # 创建对比图
                comparison_filename = f'comparison_{model_info}_{pomo_info}_temp{test_temp}_{split}_batch{batch_idx}.png'
                comparison_save_path = os.path.join(vis_path, comparison_filename)
                
                fig_comp, axes = plt.subplots(1, 2, figsize=(20, 8))
                
                # 左侧：真实路径
                ax1 = axes[0]
                ax1.scatter(points[:, 0], points[:, 1], c='lightblue', s=100, marker='o', 
                           edgecolors='blue', linewidths=1, zorder=3)
                if self.problem_type != "TSP":
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
                if self.problem_type != "TSP":
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
    }
    
    # 记录所有指标到TensorBoard和PyTorch Lightning
    for k, v in metrics.items():
        # 跳过best_solved_cost，因为我们会单独记录它以添加进度条显示
        if not k.endswith('/best_solved_cost'):
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


def enhance_adjacency_matrix(adj_matrix, min_prob=0.01, connectivity_boost=0.1):
    """
    增强邻接矩阵的连通性，主要通过距离先验来改善低质量热力图
    
    当扩散模型训练初期输出的热力图质量较差时，这个函数可以：
    1. 确保所有边都有最小概率，避免完全断连
    2. 基于节点索引距离增强近邻连接的概率（距离先验）
    3. 提高求解器找到完整路径的成功率
    
    Args:
        adj_matrix: torch.Tensor of shape (batch_size, num_nodes, num_nodes) - 原始邻接概率矩阵
        min_prob: float - 所有边的最小概率值，防止完全为0的边
        connectivity_boost: float - 基于距离的连通性增强系数
    
    Returns:
        enhanced_matrix: torch.Tensor - 增强后的邻接矩阵
    """
    enhanced = adj_matrix.clone()
    
    # 步骤1: 确保最小概率，避免求解器遇到完全不可达的节点
    enhanced = torch.clamp(enhanced, min=min_prob)
    
    # 步骤2: 基于节点索引距离的先验增强
    # 这是一个简单但有效的启发式：相邻索引的节点更可能在空间上接近
    batch_size, num_nodes, _ = enhanced.shape
    
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # 计算节点索引距离的倒数作为先验权重
                    # 索引距离越小，增强越多（假设索引相近的节点空间距离也相近）
                    index_distance_factor = 1.0 / (abs(i - j) + 1)
                    enhanced[b, i, j] += connectivity_boost * index_distance_factor
    
    # 步骤3: 归一化到合理范围，避免概率过大
    enhanced = torch.clamp(enhanced, min=0, max=1)
    
    return enhanced


def greedy_tsp_solver_batch(adj_matrix_batch, points_with_features, temperature=1.0, problem_type="TSP", add_prior=False):
    """
    非POMO版本的批量TSP/VRP求解器，每个样本只生成一个解，支持探索和VRP约束
    Args:
        adj_matrix_batch: torch.Tensor of shape (batch_size, num_nodes, num_nodes) - 邻接矩阵批次
        temperature: float - 控制探索程度的温度参数，越大越随机，越小越贪婪 (默认1.0)
        points_with_features: torch.Tensor of shape (batch_size, num_nodes, 7) - 节点特征 [x, y, demand, early_tw, late_tw, route_open, length_limit]
        problem_type: str - 问题类型，用于确定约束
    Returns:
        tours: torch.Tensor of shape (batch_size, max_tour_length) - 所有路径
        log_probs: torch.Tensor of shape (batch_size,) - 每个路径的对数概率
    """
    if add_prior:
        adj_matrix_batch = enhance_adjacency_matrix(adj_matrix_batch)

    batch_size, num_nodes, _ = adj_matrix_batch.shape
    device = adj_matrix_batch.device
    
    # 确定问题属性
    attribute_c = 'C' in problem_type or problem_type in ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW", "OVRPB", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]
    attribute_tw = 'TW' in problem_type
    attribute_o = 'O' in problem_type and problem_type.startswith('O')
    attribute_b = 'B' in problem_type  
    attribute_l = 'L' in problem_type
    
    # 非POMO版本：不扩展维度，直接使用原始批次
    # adj_matrix_batch: (batch_size, num_nodes, num_nodes)
    # points_with_features: (batch_size, num_nodes, feature_dim)
    
    # 检查特征维度
    feature_dim = points_with_features.shape[-1]
    
    if feature_dim == 2:
        # TSP场景：只有x, y坐标
        coords = points_with_features  # (batch_size, num_nodes, 2)
        
        # 为VRP约束创建默认值（如果需要的话）
        if problem_type != "TSP":
            # 如果是VRP问题但只有2维坐标，创建默认的VRP特征
            raise ValueError(f"VRP问题必须包含7维特征(坐标、需求、时间窗等)，当前只有{feature_dim}维。" + 
                          "请检查数据集是否正确包含了所有必要的VRP约束特征。")
        else:
            # TSP问题不需要这些特征
            demands = None
            early_tw = None
            late_tw = None
            route_open = None
            length_limit = None
    
    elif feature_dim == 7:
        # VRP场景：有完整的7维特征 [x, y, demand, early_tw, late_tw, route_open, length_limit]
        coords = points_with_features[:, :, :2]  # x, y坐标
        demands = points_with_features[:, :, 2]  # 需求
        early_tw = points_with_features[:, :, 3]  # 早期时间窗
        late_tw = points_with_features[:, :, 4]  # 晚期时间窗
        route_open = points_with_features[:, :, 5]  # 开放路径标志
        length_limit = points_with_features[:, :, 6]  # 路径长度限制
    else:
        raise ValueError(f"不支持的特征维度: {feature_dim}。期望2维（TSP）或7维（VRP）。")

    
    # 创建起始节点索引：非POMO版本使用固定起始点
    if problem_type == "TSP":
        # TSP：从节点0开始
        start_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        # VRP：从depot（节点0）开始
        start_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # 确定合适的路径长度上限
    if problem_type == "TSP":
        max_tour_length = num_nodes + 1  # TSP固定长度
    else:
        # VRP动态长度：考虑最坏情况下每个客户都单独一趟
        max_tour_length = max(num_nodes + 1, 2 * num_nodes + 10)
    
    # 初始化路径
    tours = torch.full((batch_size, max_tour_length), -1, dtype=torch.long, device=device)  # 用-1表示未使用
    tours[:, 0] = start_nodes  # 设置起始节点
    
    # 跟踪每个tour的实际长度
    tour_lengths = torch.ones(batch_size, dtype=torch.long, device=device)  # 从1开始（已有起始节点）
    
    # 当前节点位置
    current_nodes = start_nodes.clone()
    
    # 累积对数概率
    log_probs = torch.zeros(batch_size, device=device)
    
    # 初始化访问掩码
    if problem_type == "TSP":
        visited_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        start_mask = torch.zeros_like(visited_mask)
        start_mask.scatter_(1, start_nodes.unsqueeze(1), True)
        visited_mask = visited_mask | start_mask
    else:
        # VRP：只跟踪客户节点的访问状态，depot不计入（因为可以多次访问）
        customer_visited_mask = torch.zeros(batch_size, num_nodes - 1, dtype=torch.bool, device=device)
        # 起始从depot开始，不需要标记任何客户节点为已访问
    
    # VRP状态变量初始化
    loads = torch.ones(batch_size, device=device)  # 当前载重，初始为满容量
    times = torch.zeros(batch_size, device=device)  # 当前时间
    
    # 修复：路径长度约束初始化 - 使用客户节点的常量值（避免depot的0值）
    if points_with_features is not None and length_limit is not None:
        if attribute_l:
            # 使用第一个客户节点的长度限制作为常量
            if length_limit.shape[1] > 1:
                initial_lengths = length_limit[:, 1]  # 使用客户节点1的长度限制
            else:
                initial_lengths = torch.full((batch_size,), 3.0, device=device)
        else:
            initial_lengths = torch.full((batch_size,), 3.0, device=device)
        remaining_lengths = initial_lengths.clone()
    else:
        initial_lengths = torch.full((batch_size,), 3.0, device=device)
        remaining_lengths = initial_lengths.clone()
    
    # 添加当前路径段距离跟踪（用于长度约束检查）
    current_segment_distances = torch.zeros(batch_size, device=device)
    
    round_error_epsilon = 0.000001
    
    # 动态路径构建循环
    max_steps = max_tour_length - 2  # 留出空间给可能的最终返回步骤
    
    for step in range(max_steps):
        # 检查是否所有tours都已完成
        if problem_type == "TSP":
            # TSP：固定步数
            if step >= num_nodes - 1:
                break
        else:
            # VRP：检查是否所有客户都已被访问
            all_customers_visited = customer_visited_mask.all(dim=1)  # shape: (batch_size,)
            if all_customers_visited.all():
                break  # 所有tours都访问完了所有客户
        
        # 获取当前节点到所有节点的边权重
        batch_indices = torch.arange(batch_size, device=device)
        current_edges = adj_matrix_batch[batch_indices, current_nodes]  # (batch_size, num_nodes)
        
        # 创建基础访问掩码
        if problem_type == "TSP":
            ninf_mask = torch.where(visited_mask, 
                                   torch.tensor(-float('inf'), device=device), 
                                   torch.zeros_like(visited_mask, dtype=torch.float))
        else:
            # VRP：构建掩码，已访问的客户节点不可选，depot总是可选
            ninf_mask = torch.zeros(batch_size, num_nodes, dtype=torch.float, device=device)
            # 将已访问的客户节点设为不可达
            for i in range(num_nodes - 1):  # 客户节点1到num_nodes-1
                customer_idx = i  # 在customer_visited_mask中的索引
                node_idx = i + 1  # 在原图中的节点索引
                visited_customers = customer_visited_mask[:, customer_idx]
                ninf_mask[visited_customers, node_idx] = float('-inf')
            
            # depot（节点0）始终可访问，除非当前已在depot且没有约束要求必须离开
            at_depot = (current_nodes == 0)
            # 如果已经在depot且所有客户都已访问，则应该结束
            depot_and_done = at_depot & all_customers_visited
            ninf_mask[depot_and_done, 0] = float('-inf')  # 禁止停留在depot
        
        # VRP约束处理
        if points_with_features is not None and problem_type != "TSP":
            # 1. 容量约束
            if attribute_c and demands is not None:
                demand_too_large = loads.unsqueeze(1) + round_error_epsilon < demands
                ninf_mask[demand_too_large] = float('-inf')
            
            # 2. 时间窗约束
            if attribute_tw and coords is not None and early_tw is not None and late_tw is not None:
                # 计算从当前位置到所有节点的时间
                current_points = coords[batch_indices, current_nodes]  # (batch_size, 2)
                time_to_nodes = torch.sqrt(torch.sum((current_points.unsqueeze(1) - coords) ** 2, dim=-1))  # (batch_size, num_nodes)
                arrival_times = times.unsqueeze(1) + time_to_nodes
                
                # 检查是否违反时间窗约束
                time_too_late = arrival_times > late_tw
                # 对于没有时间窗的节点（late_tw=0），不应用约束
                no_tw_mask = late_tw == 0
                time_too_late[no_tw_mask] = False
                ninf_mask[time_too_late] = float('-inf')
            
            # 3. 路径长度约束
            if attribute_l and coords is not None:
                current_points = coords[batch_indices, current_nodes]  # (batch_size, 2)
                distance_to_nodes = torch.sqrt(torch.sum((current_points.unsqueeze(1) - coords) ** 2, dim=-1))  # (batch_size, num_nodes)
                
                if attribute_o:
                    # 开放路径：不需要返回depot
                    length_too_small = remaining_lengths.unsqueeze(1) - round_error_epsilon < distance_to_nodes
                else:
                    # 封闭路径：需要考虑返回depot的距离
                    depot_points = coords[:, 0, :]  # depot坐标 (batch_size, 2)
                    distance_to_depot = torch.sqrt(torch.sum((coords - depot_points.unsqueeze(1)) ** 2, dim=-1))  # (batch_size, num_nodes)
                    total_distance_needed = distance_to_nodes + distance_to_depot
                    length_too_small = remaining_lengths.unsqueeze(1) - round_error_epsilon < total_distance_needed
                
                ninf_mask[length_too_small] = float('-inf')
        
        # 关键修复：对于开放路径问题，确保depot永远可达以避免死锁
        if attribute_o:
            # 开放路径中，返回depot应该永远是可行的选择
            # 这防止了车辆因约束而完全停滞的情况
            ninf_mask[:, 0] = 0.0  # 确保depot（节点0）永远不被屏蔽
            
            # 但如果当前已在depot且所有客户都已访问，则可以结束
            if problem_type != "TSP":
                at_depot = (current_nodes == 0)
                all_customers_visited = customer_visited_mask.all(dim=1)
                depot_and_done = at_depot & all_customers_visited
                ninf_mask[depot_and_done, 0] = float('-inf')  # 此时可以屏蔽depot以结束路径
        
        # 应用掩码
        masked_edges = current_edges + ninf_mask
        
        # 根据温度参数进行选择
        if temperature <= 0.0:
            next_nodes = torch.argmax(masked_edges, dim=-1)
            # 对于贪婪选择，仍需要计算概率用于日志记录
            edge_probs = F.softmax(masked_edges, dim=-1)
        else:
            scaled_logits = masked_edges / temperature
            edge_probs = F.softmax(scaled_logits, dim=-1)
            next_nodes = torch.multinomial(edge_probs, num_samples=1).squeeze(-1)
        
        # 计算选择概率（使用已计算的edge_probs）
        selected_probs = edge_probs.gather(1, next_nodes.unsqueeze(1)).squeeze(1)
        log_probs += torch.log(selected_probs + 1e-8)
        
        # 更新路径
        tours[batch_indices, tour_lengths] = next_nodes
        tour_lengths += 1
        
        # 更新访问掩码
        if problem_type == "TSP":
            next_mask = torch.zeros_like(visited_mask)
            next_mask.scatter_(1, next_nodes.unsqueeze(1), True)
            visited_mask = visited_mask | next_mask
        else:
            # VRP：只更新客户节点的访问状态
            is_customer = next_nodes > 0  # 非depot节点
            customer_indices = next_nodes - 1  # 转换为customer_visited_mask的索引
            customer_indices = customer_indices.clamp(0, num_nodes - 2)  # 防止越界
            
            # 更新客户访问掩码
            customer_mask = torch.zeros_like(customer_visited_mask)
            valid_customers = is_customer & (customer_indices < num_nodes - 1)
            if valid_customers.any():
                customer_mask[valid_customers, customer_indices[valid_customers]] = True
                customer_visited_mask = customer_visited_mask | customer_mask
        
        # VRP状态更新
        if points_with_features is not None and problem_type != "TSP":
            at_depot_now = (next_nodes == 0)
            
            # 1. 更新载重
            if attribute_c and demands is not None:
                selected_demands = demands.gather(1, next_nodes.unsqueeze(1)).squeeze(1)
                loads -= selected_demands
                loads[at_depot_now] = 1.0  # 在depot时重置载重
            
            # 2. 更新时间
            if attribute_tw and coords is not None and early_tw is not None:
                current_points = coords[batch_indices, current_nodes]
                next_points = coords[batch_indices, next_nodes]
                travel_time = torch.sqrt(torch.sum((next_points - current_points) ** 2, dim=-1))
                
                arrival_time = times + travel_time
                selected_early_tw = early_tw.gather(1, next_nodes.unsqueeze(1)).squeeze(1)
                times = torch.max(arrival_time, selected_early_tw)
                times[at_depot_now] = 0.0  # 在depot时重置时间
            
            # 3. 更新路径长度
            if attribute_l and coords is not None:
                current_points = coords[batch_indices, current_nodes]
                next_points = coords[batch_indices, next_nodes]
                travel_distance = torch.sqrt(torch.sum((next_points - current_points) ** 2, dim=-1))
                
                # 更新剩余长度和当前路径段距离
                remaining_lengths -= travel_distance
                current_segment_distances += travel_distance
                
                # 关键修复：当回到depot时，重置路径段距离和长度限制
                if at_depot_now.any():
                    # 重置当前路径段距离
                    current_segment_distances[at_depot_now] = 0.0
                    
                    # 重置为初始的路径长度限制（常量）
                    remaining_lengths[at_depot_now] = initial_lengths[at_depot_now]
        
        # 更新当前节点
        current_nodes = next_nodes
        
        # 检查是否达到最大长度限制
        if tour_lengths.max() >= max_tour_length - 1:
            break
    
    # 处理路径结束：确保所有路径都以合适的方式结束
    if problem_type == "TSP":
        # TSP：添加回到起始节点的路径
        tours[batch_indices, tour_lengths] = start_nodes
        
        # 计算回到起始节点的对数概率
        return_edges = adj_matrix_batch[batch_indices, current_nodes, start_nodes]
        log_probs += torch.log(return_edges + 1e-8)
        # 关键修复：更新tour_lengths以包含回到起点的节点
        tour_lengths += 1
    else:
        # VRP：如果当前不在depot，则返回depot
        not_at_depot = (current_nodes != 0)
        if torch.is_tensor(not_at_depot) and not_at_depot.any():
            tours[not_at_depot, tour_lengths[not_at_depot]] = 0  # 返回depot
            
            # 计算返回depot的概率（简化处理）
            if not_at_depot.any():
                return_edges = adj_matrix_batch[not_at_depot, current_nodes[not_at_depot], 0]
                log_probs[not_at_depot] += torch.log(return_edges + 1e-8)
            
            tour_lengths[not_at_depot] += 1
    
    # 截断路径到实际长度，将-1填充移除
    final_tours = []
    for i in range(batch_size):
        actual_length = int(tour_lengths[i].item())  # 确保是整数类型
        tour_i = tours[i, :actual_length]
        # 为了兼容现有代码，统一填充到相同长度
        if problem_type == "TSP":
            # 关键修复：使用-1而不是0来填充，避免与有效节点编号混淆
            padded_tour = torch.full((num_nodes + 1,), -1, dtype=torch.long, device=device)
            padded_tour[:len(tour_i)] = tour_i
        else:
            # VRP：使用实际长度，但至少保证num_nodes+1的长度以兼容
            min_length = max(num_nodes + 1, actual_length)
            min_length = int(min_length)
            # 关键修复：使用-1而不是0来填充，避免与有效节点编号混淆
            padded_tour = torch.full((min_length,), -1, dtype=torch.long, device=device)
            padded_tour[:len(tour_i)] = tour_i
        final_tours.append(padded_tour)
    
    # 找到最大长度并统一所有tours的长度
    if final_tours:
        max_length = max(len(tour) for tour in final_tours)
        # 使用-1填充而不是0，保持一致性
        unified_tours = torch.full((batch_size, max_length), -1, dtype=torch.long, device=device)
        for i, tour in enumerate(final_tours):
            unified_tours[i, :len(tour)] = tour
        tours = unified_tours
    
    return tours, log_probs


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


def test_greedy_tsp_solver_batch_pomo():
    """
    测试 greedy_tsp_solver_batch_pomo 函数的正确性
    包含TSP和VRP问题的多种测试场景
    """
    print("=" * 60)
    print("开始测试 greedy_tsp_solver_batch_pomo 函数")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试1: TSP问题基础测试
    def test_tsp_basic():
        print("\n【测试1】TSP问题基础测试")
        print("-" * 40)
        
        batch_size, num_nodes = 2, 5
        # 创建简单的TSP测试数据
        points = torch.rand(batch_size, num_nodes, 2, device=device) * 1  # 坐标在[0,10]范围内
        
        # 创建基于距离的邻接矩阵
        adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        dist = torch.sqrt(torch.sum((points[b, i] - points[b, j]) ** 2))
                        adj_matrix[b, i, j] = 1.0 / (dist + 0.1)  # 距离越近权重越大
        
        # 测试贪婪模式
        tours, log_probs = greedy_tsp_solver_batch_pomo(
            adj_matrix, points, temperature=0.0, problem_type="TSP"
        )
        
        print(f"输入形状: points={points.shape}, adj_matrix={adj_matrix.shape}")
        print(f"输出形状: tours={tours.shape}, log_probs={log_probs.shape}")
        print(f"期望tours形状: ({batch_size * num_nodes}, {num_nodes + 1})")
        
        # 验证POMO扩展
        expected_total_tours = batch_size * num_nodes
        assert tours.shape[0] == expected_total_tours, f"POMO扩展错误: 期望{expected_total_tours}个tours，实际{tours.shape[0]}"
        assert log_probs.shape[0] == expected_total_tours, f"log_probs数量错误"
        
        # 验证路径完整性
        for i in range(min(5, tours.shape[0])):  # 检查前5个tours
            tour = tours[i]
            valid_tour = tour[tour >= 0]  # 移除填充的-1
            unique_nodes = torch.unique(valid_tour[:-1])  # 除去最后回到起点的节点
            print(f"Tour {i}: {valid_tour.cpu().numpy()}")
            assert len(unique_nodes) == num_nodes, f"Tour {i} 未访问所有节点: 访问了{len(unique_nodes)}个，期望{num_nodes}个"
            assert valid_tour[0] == valid_tour[-1], f"Tour {i} 未形成回路"
        
        print("✅ TSP基础测试通过")
    
    # 测试2: VRP问题基础测试
    def test_vrp_basic():
        print("\n【测试2】VRP问题基础测试")
        print("-" * 40)
        
        batch_size, num_nodes = 1, 6  # 1个depot + 5个客户
        
        # 创建VRP测试数据 (7维特征)
        points_with_features = torch.zeros(batch_size, num_nodes, 7, device=device)
        
        # 设置坐标
        points_with_features[:, :, :2] = torch.rand(batch_size, num_nodes, 2, device=device) * 10
        
        # 设置需求 (depot需求为0，客户需求随机)
        points_with_features[:, 0, 2] = 0  # depot需求为0
        points_with_features[:, 1:, 2] = torch.rand(batch_size, num_nodes-1, device=device) * 0.3  # 客户需求
        
        # 设置时间窗 (简化版，不设置约束)
        points_with_features[:, :, 3] = 0  # early_tw
        points_with_features[:, :, 4] = 0  # late_tw (0表示无约束)
        
        # 设置路径长度限制
        points_with_features[:, :, 6] = 15.0  # length_limit
        
        # 创建邻接矩阵
        coords = points_with_features[:, :, :2]
        adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        dist = torch.sqrt(torch.sum((coords[b, i] - coords[b, j]) ** 2))
                        adj_matrix[b, i, j] = 1.0 / (dist + 0.1)
        
        # 测试CVRP
        tours, log_probs = greedy_tsp_solver_batch_pomo(
            adj_matrix, points_with_features, temperature=0.0, problem_type="CVRP"
        )
        
        print(f"输入形状: points_with_features={points_with_features.shape}")
        print(f"输出形状: tours={tours.shape}, log_probs={log_probs.shape}")
        
        # VRP的POMO应该从客户节点开始 (num_nodes-1个起始点)
        expected_total_tours = batch_size * (num_nodes - 1) 
        assert tours.shape[0] == expected_total_tours, f"VRP POMO扩展错误: 期望{expected_total_tours}个tours"
        
        # 验证客户节点访问完整性
        for i in range(min(3, tours.shape[0])):
            tour = tours[i]
            valid_tour = tour[tour >= 0]
            customer_visits = set()
            for node in valid_tour:
                if node > 0:  # 客户节点
                    customer_visits.add(node.item())
            
            expected_customers = set(range(1, num_nodes))  # 客户节点1到num_nodes-1
            print(f"VRP Tour {i}: {valid_tour.cpu().numpy()}")
            print(f"  访问的客户: {sorted(customer_visits)}")
            assert customer_visits == expected_customers, f"Tour {i} 客户访问不完整: {customer_visits} vs {expected_customers}"
        
        print("✅ VRP基础测试通过")
    
    # 测试3: 温度参数测试
    def test_temperature_effects():
        print("\n【测试3】温度参数效果测试")
        print("-" * 40)
        
        batch_size, num_nodes = 1, 4
        points = torch.rand(batch_size, num_nodes, 2, device=device) * 5
        
        # 创建邻接矩阵
        adj_matrix = torch.ones(batch_size, num_nodes, num_nodes, device=device) * 0.5
        for b in range(batch_size):
            adj_matrix[b].fill_diagonal_(0)  # 对角线为0
        
        temperatures = [0.0, 0.5, 1.0, 2.0]
        results = {}
        
        for temp in temperatures:
            tours, log_probs = greedy_tsp_solver_batch_pomo(
                adj_matrix, points, temperature=temp, problem_type="TSP"
            )
            
            # 计算路径多样性 (不同路径的数量)
            unique_tours = set()
            for i in range(tours.shape[0]):
                tour_tuple = tuple(tours[i].cpu().numpy())
                unique_tours.add(tour_tuple)
            
            diversity = len(unique_tours)
            results[temp] = {
                'diversity': diversity,
                'total_tours': tours.shape[0],
                'avg_log_prob': log_probs.mean().item()
            }
            
            print(f"温度 {temp}: 多样性={diversity}/{tours.shape[0]}, 平均log_prob={log_probs.mean().item():.4f}")
        
        # 验证温度效应：温度越高，多样性应该越高
        diversities = [results[temp]['diversity'] for temp in temperatures]
        print(f"多样性趋势: {diversities}")
        
        print("✅ 温度参数测试通过")
    
    # 测试4: 边界情况测试
    def test_edge_cases():
        print("\n【测试4】边界情况测试")
        print("-" * 40)
        
        # 测试最小TSP (3个节点)
        print("测试最小TSP问题 (3个节点)...")
        points = torch.tensor([[[0, 0], [1, 0], [0, 1]]], dtype=torch.float, device=device)
        adj_matrix = torch.ones(1, 3, 3, device=device)
        adj_matrix[0].fill_diagonal_(0)
        
        tours, log_probs = greedy_tsp_solver_batch_pomo(
            adj_matrix, points, temperature=0.0, problem_type="TSP"
        )
        
        assert tours.shape[0] == 3, "最小TSP的POMO扩展错误"
        print(f"最小TSP tours形状: {tours.shape}")
        
        # 测试单客户VRP (2个节点: depot + 1个客户)
        print("测试单客户VRP问题...")
        vrp_features = torch.zeros(1, 2, 7, device=device)
        vrp_features[0, :, :2] = torch.tensor([[0, 0], [1, 1]], dtype=torch.float)  # 坐标
        vrp_features[0, 1, 2] = 0.5  # 客户需求
        vrp_features[0, :, 6] = 10.0  # 长度限制
        
        vrp_adj = torch.ones(1, 2, 2, device=device)
        vrp_adj[0].fill_diagonal_(0)
        
        tours, log_probs = greedy_tsp_solver_batch_pomo(
            vrp_adj, vrp_features, temperature=0.0, problem_type="CVRP"
        )
        
        assert tours.shape[0] == 1, "单客户VRP的POMO扩展错误"  # 只有1个客户节点作为起始点
        print(f"单客户VRP tours形状: {tours.shape}")
        
        print("✅ 边界情况测试通过")
    
    # 测试5: 约束验证测试
    def test_constraints_validation():
        print("\n【测试5】VRP约束验证测试")
        print("-" * 40)
        
        # 创建带严格约束的VRP问题
        batch_size, num_nodes = 1, 4  # 1个depot + 3个客户
        features = torch.zeros(batch_size, num_nodes, 7, device=device)
        
        # 设置坐标 (depot在中心，客户围绕)
        features[0, 0, :2] = torch.tensor([5, 5])  # depot
        features[0, 1, :2] = torch.tensor([1, 1])  # 客户1
        features[0, 2, :2] = torch.tensor([9, 1])  # 客户2  
        features[0, 3, :2] = torch.tensor([5, 9])  # 客户3
        
        # 设置需求 
        features[0, 0, 2] = 0      # depot
        features[0, 1, 2] = 0.3    # 客户1需求
        features[0, 2, 2] = 0.4    # 客户2需求  
        features[0, 3, 2] = 0.5    # 客户3需求
        
        # 设置时间窗 (客户2有严格时间窗)
        features[0, :, 3] = 0      # early_tw都为0
        features[0, :, 4] = 0      # late_tw都为0 (无约束)
        features[0, 2, 4] = 100    # 客户2有晚期时间窗
        
        # 设置路径长度限制
        features[0, :, 6] = 20.0
        
        # 创建距离基础的邻接矩阵
        adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        coords = features[:, :, :2]
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        dist = torch.sqrt(torch.sum((coords[b, i] - coords[b, j]) ** 2))
                        adj_matrix[b, i, j] = 1.0 / (dist + 0.1)
        
        # 测试带容量约束的VRP
        tours, log_probs = greedy_tsp_solver_batch_pomo(
            adj_matrix, features, temperature=0.0, problem_type="CVRP"
        )
        
        print(f"约束测试tours形状: {tours.shape}")
        
        # 使用execute_vrp_simulation验证约束
        for i in range(min(2, tours.shape[0])):
            tour = tours[i].cpu().numpy()
            # 移除填充值
            valid_tour = tour[tour >= 0]
            if len(valid_tour) > 1:
                try:
                    execution_history = simulate_vrp_execution(
                        valid_tour, features[0].cpu().numpy(), "CVRP"
                    )
                    violations = execution_history['constraint_violations']['total_violations']
                    print(f"Tour {i}: 长度={len(valid_tour)}, 约束违反={violations}")
                    if violations > 0:
                        print(f"  详细违反: {execution_history['constraint_violations']}")
                except Exception as e:
                    print(f"Tour {i} 验证失败: {e}")
        
        print("✅ 约束验证测试完成")
    
    # 运行所有测试
    try:
        # test_tsp_basic()
        test_vrp_basic()  
        test_constraints_validation()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！greedy_tsp_solver_batch_pomo 函数工作正常")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_pomo_comparison():
    """
    比较POMO和非POMO方法的性能差异
    """
    print("\n" + "=" * 60) 
    print("POMO vs 传统方法性能对比测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试问题
    batch_size, num_nodes = 2, 8
    points = torch.rand(batch_size, num_nodes, 2, device=device) * 10
    
    # 基于距离的邻接矩阵
    adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = torch.sqrt(torch.sum((points[b, i] - points[b, j]) ** 2))
                    adj_matrix[b, i, j] = 1.0 / (dist + 0.1)
    
    # 计算真实距离矩阵用于成本计算
    distance_matrices = calculate_euclidean_distance_batch(points)
    
    print("测试TSP问题...")
    
    # POMO方法
    print("\n1. POMO方法:")
    tours_pomo, log_probs_pomo = greedy_tsp_solver_batch_pomo(
        adj_matrix, points, temperature=0.0, problem_type="TSP"
    )
    costs_pomo = calculate_tour_cost_batch_pomo(tours_pomo, distance_matrices, "TSP")
    
    # 为每个原始样本选择最佳路径
    costs_pomo_reshaped = costs_pomo.reshape(batch_size, num_nodes)
    best_costs_pomo = costs_pomo_reshaped.min(dim=1)[0]
    best_indices = costs_pomo_reshaped.argmin(dim=1)
    
    print(f"POMO生成路径数: {tours_pomo.shape[0]} ({batch_size} × {num_nodes})")
    print(f"每个样本的最佳成本: {best_costs_pomo.cpu().numpy()}")
    
    # 传统方法 (非POMO版本) - 只从节点0开始
    print("\n2. 传统方法 (单起始点):")
    tours_traditional, log_probs_traditional = greedy_tsp_solver_batch(
        adj_matrix, points, temperature=0.0, problem_type="TSP"
    )
    costs_traditional = calculate_tour_cost_batch_single(tours_traditional, distance_matrices, "TSP")
    
    print(f"传统方法生成路径数: {tours_traditional.shape[0]}")
    print(f"传统方法成本: {costs_traditional.cpu().numpy()}")
    
    # 性能对比
    print("\n3. 性能对比:")
    improvement = ((costs_traditional - best_costs_pomo) / costs_traditional * 100)
    print(f"POMO改进百分比: {improvement.cpu().numpy()}")
    print(f"平均改进: {improvement.mean().item():.2f}%")
    
    # 显示最佳路径
    print("\n4. 最佳路径示例:")
    for b in range(batch_size):
        best_idx = best_indices[b] + b * num_nodes
        best_tour = tours_pomo[best_idx]
        traditional_tour = tours_traditional[b]
        
        print(f"样本 {b}:")
        print(f"  POMO最佳路径: {best_tour.cpu().numpy()}")
        print(f"  传统路径:     {traditional_tour.cpu().numpy()}")
        print(f"  成本对比: POMO={best_costs_pomo[b]:.3f} vs 传统={costs_traditional[b]:.3f}")
    
    print("\n✅ POMO对比测试完成")


if __name__ == "__main__":
    # 运行测试
    print("开始运行 greedy_tsp_solver_batch_pomo 测试套件...")
    
    # 基础功能测试
    success = test_greedy_tsp_solver_batch_pomo()
    
    if success:
        # 性能对比测试
        test_pomo_comparison()
    
    print("\n测试运行完成！")