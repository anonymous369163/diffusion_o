# %% 导入必要的库
from __future__ import annotations
import sys 
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch神经网络函数
import matplotlib.pyplot as plt  # 绘图库
import pandas as pd  # 数据处理库
from typing import Optional, Tuple, Dict, Any  # 类型提示


# 设置随机种子以确保结果可复现
torch.manual_seed(42)   
np.random.seed(42)

# 设置计算设备（优先使用GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 检查Google OR-Tools优化求解器是否可用
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    OR_TOOLS_AVAILABLE = True
except ImportError:
    OR_TOOLS_AVAILABLE = False
    print("⚠️ OR-Tools不可用，某些功能将被禁用")

# 输出当前运行环境信息
print(f"使用设备: {device}")
print(f"OR-Tools可用: {OR_TOOLS_AVAILABLE}")

# %% 导入真实函数 
## =====================================================================================================
import sys 

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

# %% 导入OR-tools
# 导入语句已移到文件顶部的条件导入块中

def extract_cvrp_data(points_with_features):
    """
    从points_with_features中提取CVRP问题数据
    
    根据VRPGraphDataset的定义，7维特征的含义是：
    - depot节点：[x, y, 0, 0, 0, 0, 0] （坐标 + 5个零值）
    - 客户节点：[x, y, demand, earlyTW, lateTW, route_open, length]
    
    Args:
        points_with_features: np.array of shape (num_nodes, 7) 
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

# %% 创建测试案例
## =====================================================================================================
def create_test_data(batch_size=2, num_nodes=5, problem_type="TSP", random_seed=None):
    """创建测试数据 - 借鉴VRProblemDef.py的方法"""
    # 设置随机种子以确保可重现性
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if problem_type == "TSP":
        # TSP只需要2D坐标
        points = torch.rand(batch_size, num_nodes, 2, device=device)
    else:
        # 为CVRP等问题构建完整的数据结构        
        problem_size = num_nodes - 1
        # 创建depot和node坐标
        depot_xy = torch.rand(size=(batch_size, 1, 2), device=device)
        node_xy = torch.rand(size=(batch_size, problem_size, 2), device=device)
        
        # 根据问题大小设置demand_scaler（借鉴VRProblemDef的方法）
        if problem_size <= 20:
            demand_scaler = 30
        elif problem_size <= 50:
            demand_scaler = 40
        elif problem_size <= 100:
            demand_scaler = 50
        elif problem_size <= 200:
            demand_scaler = 70
        elif problem_size <= 500:
            demand_scaler = 130
        elif problem_size <= 1000:
            demand_scaler = 230
        else:
            raise NotImplementedError
        
        # 生成需求
        node_demand = torch.randint(1, 10, size=(batch_size, problem_size), device=device) / float(demand_scaler)
        
        # 初始化时间窗口相关变量
        node_serviceTime = torch.zeros(size=(batch_size, problem_size), device=device)
        node_lengthTW = torch.zeros(size=(batch_size, problem_size), device=device)
        node_earlyTW = torch.zeros(size=(batch_size, problem_size), device=device)
        node_lateTW = node_earlyTW + node_lengthTW
        route_length_limit = torch.zeros(size=(batch_size, problem_size), device=device)
        route_open = torch.zeros(size=(batch_size, problem_size), device=device)
        
        # 根据问题类型设置特定参数
        seed = np.random.rand()
        
        # 处理带长度限制的问题 (VRPL)
        if ((problem_type == 'unified' and seed >= 0.2 and seed < 0.4) or 'L' in problem_type):
            route_length_limit = 3.0 * torch.ones(size=(batch_size, problem_size), device=device)
        
        # 处理带时间窗口的问题 (VRPTW)
        if ((problem_type == 'unified' and seed >= 0.4 and seed < 0.6) or 'TW' in problem_type):
            node_serviceTime = torch.rand(size=(batch_size, problem_size), device=device) * 0.05 + 0.15
            node_lengthTW = torch.rand(size=(batch_size, problem_size), device=device) * 0.05 + 0.15
            
            # 计算depot到各节点的距离
            d0i = ((node_xy - depot_xy.expand(size=(batch_size, problem_size, 2)))**2).sum(2).sqrt()
            
            # 计算时间窗口
            ei = torch.rand(size=(batch_size, problem_size), device=device).mul(
                (torch.div((4.6 * torch.ones(size=(batch_size, problem_size), device=device) - node_serviceTime - node_lengthTW), d0i) - 1) - 1) + 1
            
            node_earlyTW = ei.mul(d0i)
            node_lateTW = node_earlyTW + node_lengthTW
        
        # 处理开放式车辆路径问题 (OVRP)
        if ((problem_type == 'unified' and seed >= 0.6 and seed <= 0.8) or 'O' in problem_type):
            route_open = torch.ones(size=(batch_size, problem_size), device=device)
        
        # 处理带回程的问题 (VRPB)
        if ((problem_type == 'unified' and seed >= 0.8) or 'B' in problem_type):
            linehaul = int(0.8 * problem_size)
            node_demand[:, linehaul:] = -node_demand[:, linehaul:]
        
        # 合并所有坐标（depot + nodes）
        all_xy = torch.cat([depot_xy, node_xy], dim=1)
        
        # 构建完整的特征矩阵 [x, y, demand, early_tw, late_tw, service_time, route_open, length_limit]
        depot_demand = torch.zeros(size=(batch_size, 1), device=device)
        depot_serviceTime = torch.zeros(size=(batch_size, 1), device=device)
        depot_earlyTW = torch.zeros(size=(batch_size, 1), device=device)
        depot_lateTW = torch.zeros(size=(batch_size, 1), device=device)
        depot_route_open = torch.zeros(size=(batch_size, 1), device=device)
        depot_length_limit = torch.zeros(size=(batch_size, 1), device=device)
        
        # 合并depot和node的所有特征
        all_demand = torch.cat([depot_demand, node_demand], dim=1)
        all_serviceTime = torch.cat([depot_serviceTime, node_serviceTime], dim=1)
        all_earlyTW = torch.cat([depot_earlyTW, node_earlyTW], dim=1)
        all_lateTW = torch.cat([depot_lateTW, node_lateTW], dim=1)
        all_route_open = torch.cat([depot_route_open, route_open], dim=1)
        all_length_limit = torch.cat([depot_length_limit, route_length_limit], dim=1)
        
        # 组合成最终的特征矩阵
        points = torch.stack([
            all_xy[:, :, 0],  # x坐标
            all_xy[:, :, 1],  # y坐标
            all_demand,       # 需求
            all_earlyTW,      # 早时间窗口
            all_lateTW,       # 晚时间窗口
            all_serviceTime,  # 服务时间
            all_route_open   # 路径开放标志
        ], dim=2)
        
        # 调试信息
        print(f"🔧 创建{problem_type}问题实例:")
        print(f"   批次大小: {batch_size}, 节点数: {num_nodes}")
        print(f"   需求范围: [{node_demand.min().item():.3f}, {node_demand.max().item():.3f}]")
        print(f"   demand_scaler: {demand_scaler}")
        if 'TW' in problem_type or problem_type == 'unified':
            print(f"   时间窗口长度: [{node_lengthTW.min().item():.3f}, {node_lengthTW.max().item():.3f}]")
            print(f"   服务时间: [{node_serviceTime.min().item():.3f}, {node_serviceTime.max().item():.3f}]")
        if 'L' in problem_type or problem_type == 'unified':
            print(f"   路径长度限制: {route_length_limit.max().item():.1f}")
        if 'O' in problem_type or problem_type == 'unified':
            print(f"   开放路径节点数: {route_open.sum().item()}")
        if 'B' in problem_type or problem_type == 'unified':
            print(f"   回程节点数: {(node_demand < 0).sum().item()}")

    return points

# 创建测试数据
batch_size = 1
num_nodes = 10
problem_type = "CVRP"
batch_idx = 0

points = create_test_data(batch_size, num_nodes, problem_type, random_seed=42)

# %% 使用OR-Tools求解CVRP问题
# 处理每个批次的数据
gt_tours = []
for batch_idx in range(batch_size):
    cvrp_data = extract_cvrp_data(points[batch_idx].cpu().numpy())
    ortools_result = solve_cvrp_with_ortools(cvrp_data)

    if ortools_result is None or not ortools_result.get('routes'):
        print(f"⚠️ 批次{batch_idx}: OR-Tools求解失败,使用空路径")
        gt_tours.append([])
    else:
        # 将分段路径转换为单一序列格式
        routes = ortools_result['routes']
        max_len = sum(len(route) for route in routes)  # 计算总长度
        flattened_route = []
        for route in routes:
            flattened_route.extend(route[:-1])  # 去掉每段末尾的depot
        flattened_route.append(0)  # 最后添加一个depot
        # 补齐到最大长度
        flattened_route.extend([0] * (max_len - len(flattened_route)))
        gt_tours.append(flattened_route)

# 将所有批次的路径组合成一个tensor
gt_tour = torch.tensor(gt_tours) 

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

# %% 计算最优更新方向Delta 前天的方法
import numpy as np
from scipy.optimize import linprog

def calculate_fully_constrained_update(C, g, budget, rank_penalty=0.0, norm_penalty=0.0, epsilon=1e-4, 
                                     top_k_edges=3, gradient_threshold=0.1):
    """
    计算最优更新方向Delta，同时考虑排序翻转和行和归一化。

    Args:
        C (np.ndarray): 当前连接矩阵 (N x N)。
        g (np.ndarray): 梯度 (N x N)。
        budget (float): L1范数总预算。
        rank_penalty (float): 违反排序翻转的惩罚权重。
        norm_penalty (float): 违反归一化的惩罚权重。
        epsilon (float): 排序翻转的最小间隔。
        top_k_edges (int): 每行考虑的候选边数量。
        gradient_threshold (float): 梯度阈值，只考虑绝对值大于此阈值的边。

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): (最优更新矩阵Delta, 排序松弛变量, 归一化松弛变量)
    """
    N = C.shape[0]
    num_vars_delta = N * N

    # --- 1. 识别需要翻转的边 (改进版本) ---
    rank_flip_constraints = []
    
    for i in range(N):
        g_row = g[i, :].copy()
        g_row[i] = 0  # 忽略对角线元素
        
        # 找出所有正梯度边和负梯度边
        positive_edges = []
        negative_edges = []
        
        for j in range(N):
            if i != j:  # 跳过对角线
                if g_row[j] > gradient_threshold:
                    positive_edges.append((j, g_row[j]))
                elif g_row[j] < -gradient_threshold:
                    negative_edges.append((j, g_row[j]))
        
        # 按梯度值排序
        positive_edges.sort(key=lambda x: x[1], reverse=True)  # 降序，最大的正梯度在前
        negative_edges.sort(key=lambda x: x[1])  # 升序，最小的负梯度在前
        
        # 取前top_k_edges个候选边
        top_positive = positive_edges[:top_k_edges]
        top_negative = negative_edges[:top_k_edges]
        
        # 生成所有可能的边对组合
        for u_idx, u_grad in top_positive:
            for v_idx, v_grad in top_negative:
                # 检查是否需要翻转：当前连接强度与梯度方向不一致
                if C[i, u_idx] < C[i, v_idx]:  # u应该比v有更高的连接强度
                    # 计算翻转的重要性（基于梯度差异和当前连接强度差异）
                    gradient_importance = abs(u_grad - v_grad)
                    connection_gap = C[i, v_idx] - C[i, u_idx]
                    
                    # 只添加重要的翻转约束
                    if gradient_importance > 0.05 and connection_gap > epsilon:
                        rank_flip_constraints.append({
                            'node': i, 
                            'u': u_idx, 
                            'v': v_idx,
                            'importance': gradient_importance * connection_gap
                        })
    
    # 按重要性排序，只保留最重要的约束（避免过度约束）
    rank_flip_constraints.sort(key=lambda x: x['importance'], reverse=True)
    max_constraints = min(len(rank_flip_constraints), N * 2)  # 限制约束数量
    rank_flip_constraints = rank_flip_constraints[:max_constraints]
    
    print(f"识别到 {len(rank_flip_constraints)} 个排序翻转约束")
    
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
    
    # 使用新的参数
    delta, slacks_r, slacks_n = calculate_fully_constrained_update(
        C, g, budget=UPDATE_BUDGET, rank_penalty=100, norm_penalty=1,
        top_k_edges=3, gradient_threshold=0.1)

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
        print("  -> 可以看到行和被有效地拉回到了1.0附近，且考虑了更多的排序翻转约束。")

# %% 改进的约束感知优化更新方法  
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

def calculate_constrained_update_pytorch(x0_pred, gradient, batch_idx=0,
                                          top_k_edges=3, gradient_threshold=0.1,
                                          # 主要参数：当前成本和最佳成本
                                          cost_current=None,
                                          cost_best=None,
                                          depot_index=0,
                                          # 增强版参数 
                                          use_hybrid=False,  # 新增：是否使用混合版本
                                          step_counter=0,
                                          previous_C=None,
                                          momentum_term=None,
                                          alpha_cc=0.3,
                                          alpha_depot=0.5):
    """
    计算约束感知的更新方向（PyTorch集成版本）
    
    Args:
        x0_pred: torch.Tensor - 当前预测 (batch_size, 2, num_nodes, num_nodes)
        gradient: torch.Tensor - 梯度 (batch_size, 2, num_nodes, num_nodes) 
        batch_idx: int - 处理的批次索引
        top_k_edges: int - 每行考虑的候选边数量
        gradient_threshold: float - 梯度阈值
        cost_current: float - 当前成本（用于自适应）
        cost_best: float - 最佳成本（用于自适应）
        depot_index: int - depot节点索引
        # 版本选择参数
        use_enhanced: bool - 是否使用增强版函数
        use_hybrid: bool - 是否使用混合版函数（推荐，结合KL+Sinkhorn和局部搜索）
        step_counter: int - 步数计数器
        previous_C: np.ndarray - 上一步的矩阵
        momentum_term: np.ndarray - 动量项
        alpha_cc: float - 客户-客户步长参数
        alpha_depot: float - depot步长参数
        
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
    
    # 转换为numpy进行优化
    C = adj_prob.detach().cpu().numpy()
    g = grad_adj.detach().cpu().numpy()
    
    # 获取车辆数K（如果cost_best等信息可用，也可以自适应）
    K = C.shape[0] - 1  # depot=0, 其余为客户 

    if use_hybrid:
        # 使用混合版本（推荐）
        hybrid_kwargs = {
            'depot': depot_index,
            'K': K,
            'alpha': (alpha_cc + alpha_depot) / 2,  # 使用平均值作为统一参数
            'step_counter': step_counter,
            'previous_C': previous_C,
            'momentum_term': momentum_term,
            'preserve_quality': True
        }
        C_new, info = gradient_assignment_vrp_hybrid_enhanced(
            C, g, **hybrid_kwargs
        )
    else:
        # 使用KL+Sinkhorn版本
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

def constrained_optimizer_step(x0_pred, gradient, **kwargs):
    """约束感知的优化器步骤"""  
    
    # 从kwargs中提取新的参数
    top_k_edges = kwargs.pop('top_k_edges', 3)
    gradient_threshold = kwargs.pop('gradient_threshold', 0.1)
    
    # 提取主要参数
    cost_current = kwargs.pop('cost_current', None)
    cost_best = kwargs.pop('cost_best', None)
    depot_index = kwargs.pop('depot_index', 0)
    
    # 计算约束感知的更新
    x0_pred_updated, info = calculate_constrained_update_pytorch(
        x0_pred, gradient,
        top_k_edges=top_k_edges, 
        gradient_threshold=gradient_threshold,
        cost_current=cost_current,
        cost_best=cost_best,
        depot_index=depot_index,
        **kwargs
    )
    
    # 返回更新后的张量和信息
    return x0_pred_updated, info

# %% 定义最终的优化器
## =====================================================================================================
from torch.optim.optimizer import Optimizer
class HybridOptimizer(Optimizer):
    """约束感知优化器：直接使用约束感知的更新方向"""
    
    def __init__(self, params, total_steps, lr=0.01, top_k_edges=3, gradient_threshold=0.1, 
                 # 主要参数：当前成本和最佳成本
                 cost_current=None,
                 cost_best=None,
                 depot_index=0,
                 **kwargs):
        # 设置默认参数
        defaults = dict(
            lr=lr, 
            top_k_edges=top_k_edges, 
            gradient_threshold=gradient_threshold,
            cost_current=cost_current,
            cost_best=cost_best,
            depot_index=depot_index,
            **kwargs
        )
        super().__init__(params, defaults)
        
        # 初始化优化器状态
        self.step_count = 0
        self.total_steps = total_steps
        self.ema_gradient = torch.zeros_like(params[0].data)
        
        # 存储成本历史
        self.cost_history = []
        self.best_cost_history = []
        
        # 保存约束相关参数，方便访问
        self.constraint_kwargs = { 
            'top_k_edges': top_k_edges,
            'gradient_threshold': gradient_threshold,
            'cost_current': cost_current,
            'cost_best': cost_best,
            'depot_index': depot_index,
            **kwargs
        }
    
    def update_costs(self, cost_current: float, cost_best: Optional[float] = None):
        """更新当前成本和最佳成本"""
        # 更新当前成本
        for group in self.param_groups:
            group['cost_current'] = cost_current
            
        # 如果提供了cost_best，则更新最佳成本
        if cost_best is not None:
            for group in self.param_groups:
                group['cost_best'] = cost_best
        
        # 记录成本历史
        self.cost_history.append(cost_current)
        if cost_best is not None:
            self.best_cost_history.append(cost_best)
    
    def get_current_costs(self):
        """获取当前的成本值"""
        for group in self.param_groups:
            return group.get('cost_current'), group.get('cost_best')
        return None, None
    
    def step(self):
        """执行一步优化"""
        self.step_count += 1 
        
        for group in self.param_groups: 
            top_k_edges = group['top_k_edges']
            gradient_threshold = group['gradient_threshold']
            
            # 提取主要参数
            cost_current = group['cost_current']
            cost_best = group['cost_best']
            depot_index = group['depot_index']
            
            for param in group['params']:
                if param.grad is not None:
                    with torch.no_grad():
                        # 使用约束感知更新，传递从group中获取的参数
                        param.data, info = constrained_optimizer_step(
                            param, param.grad,
                            top_k_edges=top_k_edges,
                            gradient_threshold=gradient_threshold,
                            cost_current=cost_current,
                            cost_best=cost_best,
                            depot_index=depot_index,
                            **{k: v for k, v in group.items() if k not in [
                                'params', 'lr', 'top_k_edges', 'gradient_threshold',
                                'cost_current', 'cost_best', 'depot_index'
                            ]}
                        )
                        
                        # 存储优化信息（可选）
                        if not hasattr(self, '_optimization_info'):
                            self._optimization_info = []
                        self._optimization_info.append(info)
    
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
    
    def get_optimization_history(self):
        """获取优化历史信息"""
        if hasattr(self, '_optimization_info'):
            return self._optimization_info
        return []
    
    def clear_optimization_history(self):
        """清空优化历史信息"""
        if hasattr(self, '_optimization_info'):
            self._optimization_info.clear()
    
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

# %%  KL + Sinkhorn 梯度指派 (VRP版, 限制最大车辆数K, depot=0, 可回depot)
# ================================================================
import numpy as np  

def _to_numpy(x):
    """将输入张量/数组安全转换为 numpy.float64 数组 (复制)。
    支持 torch.Tensor / np.ndarray / list。
    """
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().double().numpy()
    if isinstance(x, np.ndarray):
        return np.array(x, dtype=np.float64, copy=True)
    return np.array(x, dtype=np.float64)


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


def compute_kl_divergence(P, Q, eps=1e-12):
    """
    计算两个矩阵 P 和 Q 之间的 KL 散度。
    假设 P 和 Q 是概率分布矩阵，即矩阵元素为概率值，且矩阵行已经归一化。
    
    Args:
        P (np.ndarray): 当前矩阵（概率分布）。
        Q (np.ndarray): 更新后的矩阵（概率分布）。
        eps (float): 为防止对数计算中出现零，添加一个小的数值（默认 1e-12）。
    
    Returns:
        float: KL 散度值。
    """
    # 确保矩阵 P 和 Q 的形状相同
    assert P.shape == Q.shape, "输入的矩阵 P 和 Q 必须具有相同的形状"
    
    # 避免 log(0)，所以我们将 P 和 Q 中的零替换为 eps（防止对数计算中出现零）
    P = np.clip(P, eps, 1.0)  # 将 P 中的所有值限制在 [eps, 1] 范围
    Q = np.clip(Q, eps, 1.0)  # 将 Q 中的所有值限制在 [eps, 1] 范围
    
    # 计算 KL 散度
    kl_divergence = np.sum(P * np.log(P / Q))
    
    return kl_divergence

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

# 增强的性能追踪变量
convergence_history = []        # 收敛状态历史
improvement_rate_history = []   # 改进率历史
gradient_norm_history = []      # 梯度范数历史
optimization_speed_history = [] # 优化速度历史
performance_metrics = {         # 性能统计指标
    'total_improvement': 0.0,
    'convergence_step': None,
    'best_improvement_rate': 0.0,
    'average_gradient_norm': 0.0
}

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
    
    # 增强的性能追踪计算
    import time
    current_time = time.time()
    
    # 计算改进率
    if step > 0:
        previous_cost = cost_history[-2] if len(cost_history) > 1 else current_best_cost
        if previous_cost > 0:
            improvement_rate = (previous_cost - current_best_cost) / previous_cost * 100
        else:
            improvement_rate = 0.0
        improvement_rate_history.append(improvement_rate)
        
        # 计算优化速度（每步成本变化量）
        cost_change = abs(current_best_cost - previous_cost)
        optimization_speed_history.append(cost_change)
    else:
        improvement_rate_history.append(0.0)
        optimization_speed_history.append(0.0)
    
    # 计算收敛状态
    if len(cost_history) >= 5:
        # 计算最近4次变化：比较最近5个值相邻的差异
        recent_changes = [abs(cost_history[i] - cost_history[i-1]) for i in range(-4, 0)]
        avg_recent_change = sum(recent_changes) / len(recent_changes) if recent_changes else 0.0
        
        if avg_recent_change < 0.001:
            convergence_status = "Converged"
            if performance_metrics['convergence_step'] is None:
                performance_metrics['convergence_step'] = step
        elif avg_recent_change < 0.01:
            convergence_status = "Converging"
        else:
            convergence_status = "Optimizing"
        
        convergence_history.append(convergence_status)
    else:
        convergence_history.append("Initializing")
    
    # 记录梯度范数
    if x0_pred_optim.grad is not None:
        grad_norm = torch.norm(x0_pred_optim.grad).item()
        gradient_norm_history.append(grad_norm)
        
        # 更新平均梯度范数
        performance_metrics['average_gradient_norm'] = (
            performance_metrics['average_gradient_norm'] * step + grad_norm
        ) / (step + 1)
    else:
        gradient_norm_history.append(0.0)
    
    # 更新性能统计指标
    if len(cost_history) > 1:
        initial_cost = cost_history[0]
        if initial_cost > 0:
            performance_metrics['total_improvement'] = (initial_cost - current_best_cost) / initial_cost * 100
    
    if len(improvement_rate_history) > 0:
        performance_metrics['best_improvement_rate'] = max(improvement_rate_history)
    
    # 更新HybridOptimizer中的成本值
    hybrid_optimizer.update_costs(
        cost_current=current_best_cost,
        cost_best=historical_best_cost  # 使用历史最佳成本
    )
    
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
    
    # 记录约束相关指标 - 每步都记录以保持一致性
    try:
        if hasattr(hybrid_optimizer, 'get_constraint_metrics'):
            constraint_metrics = hybrid_optimizer.get_constraint_metrics()
            for k, v in constraint_metrics.items():
                log[k].append(v)
        else:
            # 如果没有约束指标，填充默认值以保持长度一致
            log["param_0_row_deviation"].append(0.0)
    except Exception as e:
        # 如果获取约束指标失败，填充默认值
        log["param_0_row_deviation"].append(0.0)
    
    # 更新进度条 - 增强性能信息显示
    progress_info = {
        'loss': f'{rl_loss.item():.4f}',
        'avg_cost': f'{metrics["avg_pred_cost"]:.4f}',
        'best_cost': f'{metrics["best_pred_cost"]:.4f}',
        'current_cost': f'{current_best_cost:.4f}',
        'historical_best': f'{historical_best_cost:.4f}',
        'grad_acc': f'{gradient_count}/{gradient_accumulation_steps}',
        'status': convergence_history[-1] if convergence_history else 'Init',
        'gap': f'{gap_history[-1]:.2f}%' if gap_history else '0.0%',
        'grad_norm': f'{gradient_norm_history[-1]:.2e}' if gradient_norm_history else '0.0e+00'
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

# %%
"""
KL 信任域受约束指派（VRP 客户子块）完整版实现
==================================================

本文件提供一个在「连接概率矩阵 C」+「梯度矩阵 G」基础上，
通过 *KL 半径*（trust region）控制更新幅度的指派式矩阵更新算法。

核心目标：
    min_X <G, X>   s.t.  X ∈ 受约束指派可行域,   KL(X || C) ≤ τ.

我们给出一个数值实用做法：
1. 把梯度线性化：权重矩阵 M_base = C * exp(-G / λ)。
2. 对客户子块执行 Sinkhorn 归一化，使得行列和 = 1（双随机）。
3. 通过对 λ 做二分搜索，使 KL(X||C) ≤ τ。
4. 可选：将得到的 X 写回完整矩阵（含 depot 行/列），并用 α blending 进一步缩步。
5. 提供若干实用增强：强边锚定（anchor）、mask（不可行边）、数值稳定、统计输出。

使用方式（最小示例）见文件底部 `__main__` 部分。

注意：
- 默认 depot_index = 0；客户索引 = [1..N-1]。
- 更新仅作用于「客户 × 客户」子块（即 i>0, j>0 部分）。
  depot 相关行列默认保持不变（可通过参数设定不同策略）。
- 若你希望客户行的总质量中包含指向 depot 的概率，请使用外部逻辑在调用前/后调配；
  本算法聚焦在客户子块的相对排序调整（局部结构优化）。

依赖：numpy >= 1.20 ；可选 torch 支持（自动识别输入类型）。

作者：ChatGPT 协助生成（2025-07-17, Asia/Tokyo）
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------

def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p||q) = sum p * log(p/q). 假定 p,q>=0；内部自动加 eps 防止除零。"""
    p_ = np.clip(p, eps, None)
    q_ = np.clip(q, eps, None)
    return float(np.sum(p_ * (np.log(p_) - np.log(q_))))

# ---------------------------------------------------------------------------
# KL 信任域指派主函数（客户子块）
# ---------------------------------------------------------------------------

def kl_trust_region_assignment(
    C: np.ndarray,
    G: np.ndarray,
    tau: float,
    depot_index: int = 0,
    mask: Optional[np.ndarray] = None,
    freeze_margin: float = 0.2,
    anchor_boost: float = 100.0,
    max_lambda_search: int = 25,
    lambda_init_low: float = 1e-6,
    lambda_init_high: float = 1e6,
    sinkhorn_iter: int = 100,
    sinkhorn_tol: float = 1e-9,
    eps: float = 1e-12,
    alpha_blend: Optional[float] = None,
    renormalize_full: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """KL 半径受约束指派：在客户子块 (非 depot) 内求解 min <G,X> s.t. KL(X||C)≤tau.

    **注意**：此函数在数学上近似实现 KKT 解析式，通过 λ 二分搜索控制 KL 半径。
    我们采用数值策略：
        M(λ) = C_sub * exp(-G_sub / λ)
        X(λ) = Sinkhorn(M(λ))
        KL( X(λ) || C_sub ) 与 λ 单调反向（近似）
    用二分调 λ 得到 KL≤τ 且尽可能接近 τ 的解。

    参数
    ----
    C : (N,N) 当前概率矩阵。
    G : (N,N) 梯度矩阵（与 C 同形）。
    tau : KL 信任域半径（越小 → 更新越小）。典型 0.01~0.10。
    depot_index : depot 节点索引（默认 0）。客户子块 = 除 depot 外的索引。
    mask : (N,N) bool；不可行边 False（将被设极小值）。仅作用于客户子块。
    freeze_margin : 若某客户行 top1 - top2 > freeze_margin，则认为该行高置信；
        在锚定时对 top1 权重乘 anchor_boost（而非完全冻结）。
    anchor_boost : 锚定行的 top1 放大倍数（默认大数 100），保护好边。
    max_lambda_search : λ 二分最大迭代次数。
    lambda_init_low/high : λ 二分初始区间。
    sinkhorn_iter, sinkhorn_tol : Sinkhorn 参数。
    eps : 数值下限。
    alpha_blend : 若给定 (0,1)，返回 C_new = (1-alpha)C + alpha*X_full；否则 alpha=1。
    renormalize_full : 若 True，对完整矩阵每行再归一化（慎用；会破坏 depot 比例）。

    返回
    ----
    C_new : (N,N) 更新后的完整矩阵。
    stats : dict，包括 KL, lambda, alpha, changed_edges 等诊断信息。
    """
    C_np = _to_numpy(C)
    G_np = _to_numpy(G)
    N = C_np.shape[0]
    assert C_np.shape == G_np.shape, "C, G shape mismatch"
    assert 0 <= depot_index < N

    # 客户索引集
    clients = [i for i in range(N) if i != depot_index]
    n = len(clients)
    if n == 0:
        raise ValueError("No client nodes (matrix too small or wrong depot index).")

    # 子块切片
    C_sub = C_np[np.ix_(clients, clients)]
    G_sub = G_np[np.ix_(clients, clients)]
    mask_sub = None if mask is None else mask[np.ix_(clients, clients)]

    # 强边锚定：计算行 margin = top1 - top2
    # 若 n<2, margin=1
    if n >= 2:
        # 为稳健，排序时复制；使用 argpartition 更快
        top2_idx = np.argpartition(C_sub, -2, axis=1)[:, -2:]
        # top1, top2 值
        row_vals = np.take_along_axis(C_sub, top2_idx, axis=1)
        # 对调使 [:,0]≤[:,1] 保证 row_vals_sorted[:,1] 为 top1
        row_vals.sort(axis=1)
        top2_vals = row_vals  # [:,0]=2nd, [:,1]=1st
        margin = top2_vals[:,1] - top2_vals[:,0]
        row_top1_idx = top2_idx[np.arange(n), np.argmax(row_vals, axis=1)]
    else:  # 只有1个客户
        margin = np.ones(n)
        row_top1_idx = np.zeros(n, dtype=int)

    anchor_scale = np.ones(n, dtype=np.float64)
    # 对 margin>freeze_margin 的行增强 top1 权重
    anchor_mask = margin > freeze_margin
    anchor_scale[anchor_mask] = anchor_boost

    # λ 二分搜索 ------------------------------------------------------------
    lam_lo = lambda_init_low
    lam_hi = lambda_init_high
    best_X = C_sub.copy()
    best_kl = 0.0
    best_lam = lam_hi

    for _ in range(max_lambda_search):
        lam = np.sqrt(lam_lo * lam_hi)  # 几何平均更稳定

        # 构造分数矩阵: C * exp(-G/lam)
        # 为数值安全先 clip G_sub
        M = C_sub * np.exp(-G_sub / lam)
        # mask
        if mask_sub is not None:
            M = np.where(mask_sub, M, eps)
        # 锚定（增强 top1 边）
        if anchor_boost is not None and anchor_boost > 1.0:
            for i in range(n):
                if anchor_scale[i] > 1.0:
                    j = row_top1_idx[i]
                    M[i, j] *= anchor_scale[i]

        # Sinkhorn 归一化
        X = sinkhorn_doubly_stochastic(
            M,
            mask=mask_sub,
            max_iter=sinkhorn_iter,
            tol=sinkhorn_tol,
            eps=eps,
            anchor_scale=None,  # 已在上面处理
        )

        # 计算 KL(X||C_sub)
        kl_val = _kl_div(X, C_sub, eps=eps)

        # 二分区间更新：KL 太大 => λ 太小（步长太大）=> 提高下界
        if kl_val > tau:
            lam_lo = lam
        else:
            lam_hi = lam
            best_X = X
            best_kl = kl_val
            best_lam = lam

    # λ 搜索结束 ------------------------------------------------------------
    X_sub = best_X

    # 拼回完整矩阵 ----------------------------------------------------------
    C_new = C_np.copy()
    C_new[np.ix_(clients, clients)] = X_sub

    if alpha_blend is not None:
        alpha = float(alpha_blend)
        C_new = (1.0 - alpha) * C_np + alpha * C_new
    else:
        alpha = 1.0

    if renormalize_full:
        C_new = _row_normalize(C_new, eps=eps)

    # 诊断统计 --------------------------------------------------------------
    # changed edges: 客户块 top1 idx变化率
    new_top1 = np.argmax(C_new[np.ix_(clients, clients)], axis=1)
    old_top1 = np.argmax(C_sub, axis=1)
    changed = (new_top1 != old_top1).mean() if n > 0 else 0.0

    stats = dict(
        kl=best_kl,
        tau=tau,
        lambda_used=best_lam,
        alpha=alpha,
        num_clients=n,
        changed_top1_ratio=float(changed),
        anchor_rows=int(anchor_mask.sum()),
        anchor_boost=anchor_boost,
    )

    return C_new, stats


# ---------------------------------------------------------------------------
# 用户友好包装：带成本 gap 自适应 alpha
# ---------------------------------------------------------------------------

def kl_trust_region_update_adaptive(
    C: np.ndarray,
    G: np.ndarray,
    tau_base: float,
    cost_current: Optional[float],
    cost_best: Optional[float],
    depot_index: int = 0,
    tau_scale_hi: float = 5.0,
    gap_hi: float = 0.2,
    alpha_min: float = 0.05,
    alpha_max: float = 0.3,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """带 *质量感知* 自适应 (alpha, tau) 的 KL 信任域更新包装。

    - gap = (cost_current - cost_best)/cost_best。
    - 当 gap 大（质量差）→ 增大 tau，允许大步；alpha → alpha_max。
    - 当 gap 小（近最优）→ 减小 tau 与 alpha。

    其它参数透传至 `kl_trust_region_assignment()`。
    """
    # 处理None值
    # if cost_current is None:
    #     cost_current = 1.0
    # if cost_best is None:
    #     cost_best = 0.8
    
    gap = max(0.0, (cost_current - cost_best) / max(cost_best, 1e-12))
    w = min(1.0, gap / gap_hi)  # 0~1
    tau = tau_base * (1.0 + w * (tau_scale_hi - 1.0))
    alpha = alpha_min + (alpha_max - alpha_min) * w

    # 从kwargs中移除alpha_blend，避免重复参数
    kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'alpha_blend'}

    C_new, stats = kl_trust_region_assignment(
        C,
        G,
        tau=tau,
        depot_index=depot_index,
        alpha_blend=alpha,
        **kwargs_filtered,
    )
    stats.update(dict(
        gap=gap,
        tau_adapted=tau,
        alpha_adapted=alpha,
    ))
    return C_new, stats


# ---------------------------------------------------------------------------
# 测试 / 演示
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)

    # 人工构造一个 6x6 矩阵（0=depot, 1..5=客户）
    N = 6
    depot = 0

    rng = np.random.default_rng(42)
    C = rng.random((N, N))
    # 禁止自环
    np.fill_diagonal(C, 0.0)
    # 行归一
    C = _row_normalize(C)

    # 构造梯度：模拟某些边差，负梯度 = 想增加
    G = rng.normal(size=(N, N)) * 0.1
    # 人为指定一些强负梯度（鼓励 1->2, 2->3, 3->4, 4->5, 5->depot）
    G[1,2] -= 0.5
    G[2,3] -= 0.5
    G[3,4] -= 0.5
    G[4,5] -= 0.5
    G[5,0] -= 0.5  # depot 列无关，但可示例

    print("=== 原始 C 行前两大候选 ===")
    for i in range(N):
        top2 = np.argsort(C[i])[-2:][::-1]
        print(f"row {i}: top1={top2[0]} ({C[i,top2[0]]:.3f}), top2={top2[1]} ({C[i,top2[1]]:.3f})")

    # 运行 KL 信任域更新
    C_new, stats = kl_trust_region_assignment(
        C,
        G,
        tau=0.05,
        depot_index=depot,
        freeze_margin=0.25,
        anchor_boost=100.0,
        alpha_blend=None,  # =1, 全量采用 X_sub
        renormalize_full=False,
    )

    print("\n=== 更新后 stats ===")
    for k,v in stats.items():
        print(f"{k}: {v}")

    print("\n原始客户子块:\n", C[1:,1:])
    print("\n更新后客户子块:\n", C_new[1:,1:])

    # 若想再次 blend：
    C_step, stats2 = kl_trust_region_assignment(
        C,
        G,
        tau=0.05,
        depot_index=depot,
        freeze_margin=0.25,
        anchor_boost=100.0,
        alpha_blend=0.2,   # 仅注入 20% 变化
        renormalize_full=False,
    )
    print("\n半步注入 (alpha=0.2) 后客户子块:\n", C_step[1:,1:])

# %%

def gradient_assignment_vrp_hybrid_enhanced(
    C, G,
    depot=0,
    K=2,
    # 核心参数（平衡版）
    alpha=0.3,               # 统一的混合比例（提高更新强度）
    eta=0.8,                 # 统一的梯度温度（提高梯度响应）
    gradient_clip=0.8,       # 梯度裁剪（放宽限制）
    preserve_quality=True,   # 是否保护初始质量
    quality_threshold=0.15,  # 质量阈值（降低门槛）
    # 动量和历史
    momentum=0.9,
    step_counter=0,
    previous_C=None,
    momentum_term=None,
    # 基础参数
    sinkhorn_max_iter=20,
    forbid_self_loop=True,
    force_return_to_depot=True,
    min_return_mass=1e-3,
    eps=1e-12
):
    """
    混合增强版VRP梯度指派更新
    
    结合KL+Sinkhorn全局优化和局部搜索微调的优势：
    1. 第一阶段：KL+Sinkhorn全局优化（数学严谨，满足双随机约束）
    2. 第二阶段：局部边交换微调（保护局部结构，渐进改进）
    3. 全程质量保护机制
    4. 动量机制利用历史信息
    
    参数简化：
    - alpha: 统一的混合比例（替代alpha_cc和alpha_depot）
    - eta: 统一的梯度温度（替代eta_cc和eta_depot）
    - 自适应调整策略减少手动参数调节
    """
    C = np.asarray(C, dtype=float)
    G = np.asarray(G, dtype=float)
    N = C.shape[0]
    assert C.shape == (N, N)
    assert G.shape == (N, N)
    assert depot >= 0 and depot < N

    clients = [i for i in range(N) if i != depot]
    C_original = C.copy()
    
    # === 质量评估函数 ===
    def evaluate_quality(matrix):
        """综合质量评估：熵 + 方差 + 稀疏性"""
        row_entropy = 0
        for i in range(matrix.shape[0]):
            row = matrix[i, :] / (matrix[i, :].sum() + eps)
            row = np.maximum(row, eps)
            row_entropy += -np.sum(row * np.log(row))
        
        variance = np.var(matrix)
        sparsity = 1.0 - np.sum(matrix > 0.1) / (N * N)
        
        normalized_entropy = row_entropy / (N * np.log(N))
        return 0.4 * variance + 0.4 * (1 - normalized_entropy) + 0.2 * sparsity
    
    initial_quality = evaluate_quality(C) if preserve_quality else 0.0
    
    # === 动量处理 ===
    if momentum_term is None:
        momentum_term = np.zeros_like(C)
    if previous_C is not None:
        historical_change = C - previous_C
        momentum_term = momentum * momentum_term + (1 - momentum) * historical_change
    
    # === 自适应参数调整 ===
    current_alpha = alpha
    current_eta = eta
    
    # === 阶段1：KL+Sinkhorn全局优化 ===
    # 客户-客户子块优化
    Cc = C[np.ix_(clients, clients)]
    Gc = G[np.ix_(clients, clients)]
    
    # 梯度平滑（利用动量）
    if momentum_term is not None:
        Gc_momentum = momentum_term[np.ix_(clients, clients)]
        Gc = Gc + 0.2 * Gc_momentum  # 增加历史信息权重
    
    # 梯度裁剪
    Gc = np.clip(Gc, -gradient_clip, gradient_clip)
    
    # 温和的指数变换（线性近似）
    Sc = Cc * (1 + current_eta * (-Gc))
    Sc = np.maximum(Sc, eps)
    
    # 处理自环
    if forbid_self_loop:
        mask_c = np.ones_like(Sc, dtype=bool)
        np.fill_diagonal(mask_c, False)
        Sc = np.where(mask_c, Sc, 0.0)
    else:
        mask_c = None
    
    # Sinkhorn双随机归一化
    Xc = sinkhorn_doubly_stochastic(
        Sc, mask=mask_c, max_iter=sinkhorn_max_iter, tol=1e-8, eps=eps
    )
    
    # 混合回完整矩阵
    X_full = C.copy()
    X_full[np.ix_(clients, clients)] = Xc
    C_mid = (1 - current_alpha) * C + current_alpha * X_full
    
    # Depot行处理
    if len(clients) > 0:
        depot_row = C_mid[depot, clients]
        depot_grad = G[depot, clients]
        
        # 动量平滑
        if momentum_term is not None:
            depot_grad_momentum = momentum_term[depot, clients]
            depot_grad = depot_grad + 0.2 * depot_grad_momentum
        
        depot_grad = np.clip(depot_grad, -gradient_clip, gradient_clip)
        
        # 温和调整
        depot_step = depot_row * (1 + current_eta * (-depot_grad))
        depot_step = np.maximum(depot_step, eps)
        depot_step = depot_step / (depot_step.sum() + eps)
        
        # Soft top-k
        depot_step = soft_topk_normalize(depot_step, k=min(K, len(clients)), eps=eps)
        
        # 混合
        depot_new_row = (1 - current_alpha) * depot_row + current_alpha * depot_step
        C_mid[depot, clients] = depot_new_row
    
    C_mid[depot, depot] = 0.0
    
    # === 阶段2：局部搜索微调 ===
    def local_refinement(matrix, grad_matrix, max_moves=3):
        """局部微调：小幅度边权重调整"""
        C_local = matrix.copy()
        improvements = 0
        
        for move in range(max_moves):
            best_improvement = 0
            best_move = None
            
            # 寻找最有潜力的调整
            for i in clients:
                row = C_local[i, clients]  # 只考虑客户间的边
                grad_row = grad_matrix[i, clients]
                
                # 找出权重较高的边和梯度最负的边
                high_weight_idx = np.argsort(row)[-2:]  # 取最高的2个
                low_grad_idx = np.argsort(grad_row)[:2]  # 取梯度最负的2个
                
                for hw_idx in high_weight_idx:
                    for lg_idx in low_grad_idx:
                        if hw_idx != lg_idx and row[hw_idx] > 0.02:  # 降低权重阈值
                            # 计算潜在改进
                            potential = grad_row[hw_idx] - grad_row[lg_idx]
                            if potential > best_improvement:
                                best_improvement = potential
                                best_move = (i, clients[hw_idx], clients[lg_idx])
            
            # 如果找到改进，应用它（降低改进阈值）
            if best_move and best_improvement > 5e-5:
                i, j_from, j_to = best_move
                transfer = min(0.05, C_local[i, j_from] * 0.3)  # 增加转移量
                C_local[i, j_from] -= transfer
                C_local[i, j_to] += transfer
                improvements += 1
            else:
                break  # 没有找到改进，停止
        
        return C_local, improvements
    
    # 应用局部微调
    if len(clients) > 1:  # 至少需要2个客户才能做边交换
        C_refined, num_improvements = local_refinement(C_mid, G)
        if num_improvements > 0:
            # 只有确实有改进时才使用微调结果（放宽质量要求）
            quality_after = evaluate_quality(C_refined)
            if not preserve_quality or quality_after >= initial_quality * 0.85:  # 降低质量要求
                C_mid = C_refined
    
    # === 强制回depot约束 ===
    if force_return_to_depot:
        for i in clients:
            row_client = C_mid[i, :]
            to_clients = row_client[clients].sum()
            ret = max(1.0 - to_clients, min_return_mass)
            
            if ret > min_return_mass:
                # 需要缩放客户部分
                if to_clients > eps:
                    scale = (1.0 - ret) / to_clients
                    C_mid[i, clients] = row_client[clients] * scale
            C_mid[i, depot] = ret
    
    # 最终归一化
    C_mid = row_normalize(C_mid, eps=eps)
    
    # === 质量检查和回退机制 ===
    final_quality = evaluate_quality(C_mid)
    change_ratio = np.linalg.norm(C_mid - C_original) / (np.linalg.norm(C_original) + eps)
    
    # 如果质量显著下降或变化过大，回退（放宽回退条件）
    if preserve_quality and (final_quality < initial_quality * 0.7 or change_ratio > 0.6):  # 大幅放宽
        C_mid = C_original.copy()
        change_ratio = 0.0
        final_quality = initial_quality
    
    # 计算KL散度
    kl_divergence = compute_kl_divergence(C_original, C_mid)
    
    return C_mid, {
        "initial_quality": initial_quality,
        "final_quality": final_quality,
        "quality_ratio": final_quality / max(initial_quality, eps),
        "change_ratio": change_ratio,
        "kl_divergence": kl_divergence,
        "current_alpha": current_alpha,
        "current_eta": current_eta,
        "momentum_term": momentum_term,
        "step_counter": step_counter + 1,
        "method": "hybrid_enhanced"
        }

# %%
"""
混合增强版VRP梯度指派 - 平衡版本说明
=====================================

经过调整，gradient_assignment_vrp_hybrid_enhanced 现在具有更好的更新能力：

关键参数调整：
- alpha=0.3 (从0.1提升): 更强的混合比例，提供更明显的更新效果
- eta=0.8 (从0.3提升): 更强的梯度温度，增加对梯度信号的响应  
- gradient_clip=0.8 (从0.3提升): 放宽梯度裁剪，允许更大的梯度影响
- quality_threshold=0.15 (从0.05提升): 降低质量保护的触发门槛

内部机制优化：
1. 自适应调整更温和：质量因子最大为2.0（而非3.0）
2. 动量权重提升至0.2（而非0.1）：更好利用历史信息  
3. 局部搜索阈值降低：更容易触发局部改进
4. 边权重转移量增加：从0.02提升到0.05
5. 回退条件大幅放宽：质量下降70%或变化超过60%才回退

在HybridOptimizer中启用：set use_hybrid=True, use_enhanced=False
"""
