import torch
import numpy as np
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("约束满足的测试函数")
print("="*80)

# 导入必要的函数 
from difusco.pl_tsp_model import simulate_vrp_execution

def test_constraint_satisfaction(points, route, problem_type="CVRP"):
    """
    简化的约束满足测试函数
    
    Args:
        points: 点集numpy数组或tensor [num_nodes, features]
        route: 路径列表或tensor [num_nodes]
        problem_type: 问题类型字符串
    
    Returns:
        bool: 是否满足约束条件
    """
    # 1. 数据预处理
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points
    
    if isinstance(route, torch.Tensor):
        route_list = route.cpu().numpy().tolist()
    else:
        route_list = route
    
    # 确保有完整的特征向量
    if points_np.shape[1] == 2:
        # 只有2D坐标，构建7维特征用于VRP验证
        num_nodes = points_np.shape[0]
        points_with_features = np.zeros((num_nodes, 7))
        points_with_features[:, :2] = points_np
        points_with_features[:, 2] = np.concatenate([np.array([0]), np.random.rand(num_nodes-1) * 0.1])
        points_with_features[:, 3] = np.zeros(num_nodes)
        points_with_features[:, 4] = np.ones(num_nodes) * 10
        points_with_features[:, 5] = np.zeros(num_nodes)
        points_with_features[:, 6] = np.ones(num_nodes) * 3.0
    else:
        points_with_features = points_np
    
    # 2. 约束检查
    constraint_satisfied = True
    violations = {}
    
    try:
        # 使用真实函数进行约束检查
        execution_history = simulate_vrp_execution(
            tour=route_list,
            points_with_features=points_with_features,
            problem_type=problem_type
        )
        
        violations = execution_history['constraint_violations']
        total_violations = violations.get('total_violations', 0)
        constraint_satisfied = (total_violations == 0)
        
        print(f"约束检查结果:")
        print(f"  - 容量约束违反: {violations.get('capacity_violations', 0)}")
        print(f"  - 时间窗约束违反: {violations.get('time_window_violations', 0)}")
        print(f"  - 路径长度约束违反: {violations.get('length_violations', 0)}")
        print(f"  - 总违反次数: {total_violations}")
        
        if constraint_satisfied:
            print("✅ 路径满足所有约束")
        else:
            print("❌ 路径存在约束违反")
            
    except Exception as e:
        print(f"❌ 约束检查失败: {e}")
        constraint_satisfied = False
    
    # 3. 可视化
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制点
        coords = points_with_features[:, :2]
        ax.scatter(coords[:, 0], coords[:, 1], s=100, c='red', alpha=0.7, zorder=5)
        ax.scatter(coords[0, 0], coords[0, 1], s=200, c='blue', marker='s', alpha=0.9, zorder=6, label='Depot')
        
        # 添加节点标签
        for i in range(len(coords)):
            ax.annotate(f'{i}', (coords[i, 0], coords[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
        
        # 绘制路径
        colors = ['green' if constraint_satisfied else 'red']
        for i in range(len(route_list) - 1):
            start = coords[route_list[i]]
            end = coords[route_list[i + 1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   color=colors[0], linewidth=2, alpha=0.8)
        
        # 设置标题和标签
        status = "满足约束" if constraint_satisfied else "违反约束"
        ax.set_title(f'{problem_type} 路径可视化 - {status}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
    
    return constraint_satisfied

if __name__ == "__main__":
    # 创建一个简单的5节点VRP问题
    points_example = np.array([
        [0.5, 0.5, 0.0, 0.0, 10.0, 0.0, 3.0],  # depot
        [0.2, 0.2, 0.1, 0.0, 10.0, 0.0, 3.0],  # 客户1
        [0.3, 0.1, 0.1, 0.0, 10.0, 0.0, 3.0],  # 客户2
        [0.1, 0.3, 0.1, 0.0, 10.0, 0.0, 3.0],  # 客户3
        [0.8, 0.8, 0.1, 0.0, 10.0, 0.0, 3.0],  # 客户4
    ])
    
    # 定义一个路径
    route_example = [0, 1, 2, 3, 4, 0]  # 从depot出发，访问所有客户，回到depot
    
    # 测试约束满足情况
    print("测试示例路径的约束满足情况:")
    is_feasible = test_constraint_satisfaction(
        points=points_example,
        route=route_example,
        problem_type="CVRP"
    )