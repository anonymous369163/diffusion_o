"""
OR-TOOLS使用
"""

# 导入必要的库
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt

# 导入OR-tools
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


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

#  创建测试案例
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

# 使用OR-Tools求解CVRP问题
# 处理每个批次的数据
if __name__ == "__main__":      
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