"""
Draw utils for TSP.

"""

import numpy as np
import matplotlib.pyplot as plt

def draw_tours(tours, gt_tour, np_points):
    import matplotlib.pyplot as plt
    
    # 计算需要的子图数量
    n_tours = len(tours) + 1  # 加1是为了包含gt_tour
    
    # 计算子图的行列数
    n_cols = min(4, n_tours)  # 每行最多4个子图
    n_rows = (n_tours + n_cols - 1) // n_cols
    
    # 创建一个图形，包含多个子图
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    # 绘制所有生成的路径
    for idx, tour in enumerate(tours):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.scatter(np_points[:, 0], np_points[:, 1], c='blue', s=50)
        
        # 连接路径的点
        for i in range(len(tour)):
            start = np_points[tour[i]]
            end = np_points[tour[(i + 1) % len(tour)]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'r-', alpha=0.7, linewidth=2)
        
        plt.title(f'Generated Tour {idx+1}')
    
    # 绘制ground truth路径
    plt.subplot(n_rows, n_cols, n_tours)
    plt.scatter(np_points[:, 0], np_points[:, 1], c='blue', s=50)
    
    # 连接ground truth路径的点
    for i in range(len(gt_tour)):
        start = np_points[gt_tour[i]]
        end = np_points[gt_tour[(i + 1) % len(gt_tour)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', alpha=0.7, linewidth=2)
    
    plt.title('Ground Truth Tour')
    
    plt.tight_layout()
    plt.show()

def compute_matrix_diff(gt_matrix, gen_matrix):
    # 计算不同类型的差异数量
    correct_ones = np.sum((gt_matrix > 0.5) & (gen_matrix > 0.5))  # 正确预测为1
    correct_zeros = np.sum((gt_matrix <= 0.5) & (gen_matrix <= 0.5))  # 正确预测为0
    missed_ones = np.sum((gt_matrix > 0.5) & (gen_matrix <= 0.5))  # 漏检（应该是1但预测为0）
    false_ones = np.sum((gt_matrix <= 0.5) & (gen_matrix > 0.5))  # 误检（应该是0但预测为1）
    
    total_elements = gt_matrix.size
    total_ones_gt = np.sum(gt_matrix > 0.5)
    
    print(f"矩阵统计:")
    print(f"总元素数: {total_elements}")
    print(f"Ground Truth中1的数量: {total_ones_gt}")
    print(f"正确预测1的数量: {correct_ones}")
    print(f"正确预测0的数量: {correct_zeros}")
    print(f"漏检数量（应为1预测为0）: {missed_ones}")
    print(f"误检数量（应为0预测为1）: {false_ones}")
    print(f"准确率: {(correct_ones + correct_zeros) / total_elements:.2%}")
    print(f"召回率: {correct_ones / total_ones_gt:.2%}")
    
    return {
        "correct_ones": correct_ones,
        "correct_zeros": correct_zeros,
        "missed_ones": missed_ones,
        "false_ones": false_ones
    }

def plot_matrix(gen_matrix, gt_matrix):
    """
    可视化生成矩阵和真实矩阵的差异对比
    
    Args:
        gen_matrix: 生成的矩阵
        gt_matrix: Ground Truth矩阵
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    
    # 绘制生成矩阵
    plt.subplot(131)
    plt.imshow(gen_matrix, cmap='Blues')
    plt.colorbar()
    plt.title('生成矩阵')
    
    # 绘制真实矩阵
    plt.subplot(132) 
    plt.imshow(gt_matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Ground Truth矩阵')
    
    # 绘制差异矩阵
    plt.subplot(133)
    diff = gen_matrix - gt_matrix
    plt.imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('差异矩阵\n蓝色:漏检 红色:误检')
    
    # 计算并显示统计信息
    stats = compute_matrix_diff(gt_matrix, gen_matrix)
    plt.suptitle(
        f"正确预测1: {stats['correct_ones']}, "
        f"漏检: {stats['missed_ones']}, "
        f"误检: {stats['false_ones']}"
    )
    
    plt.tight_layout()
    plt.show()


def visualize_tsp_solutions(np_points, np_gt_tour, gt_cost, best_tour, best_real_cost, idx):
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        
        # 在WSL环境下设置合适的后端
        matplotlib.use('Agg')  # 非交互式后端，适合WSL
        
        plt.figure(figsize=(14, 6))
        
        # 计算合适的箭头大小（基于坐标范围）
        coord_range_x = np.max(np_points[:, 0]) - np.min(np_points[:, 0])
        coord_range_y = np.max(np_points[:, 1]) - np.min(np_points[:, 1])
        coord_range = max(coord_range_x, coord_range_y)
        arrow_head_width = coord_range * 0.01  # 箭头宽度为坐标范围的1%
        arrow_head_length = coord_range * 0.008  # 箭头长度为坐标范围的0.8%
        
        # 绘制真实最优解
        plt.subplot(1, 2, 1)
        plt.scatter(np_points[:, 0], np_points[:, 1], c='red', s=50, zorder=5, alpha=0.6)
        for i, (x, y) in enumerate(np_points):
            plt.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points', 
                        fontsize=8, ha='left')
        
        # 绘制真实最优路径 - 使用较小的箭头
        gt_tour_extended = list(np_gt_tour) + [np_gt_tour[0]]
        for i in range(len(gt_tour_extended) - 1):
            start_idx = gt_tour_extended[i]
            end_idx = gt_tour_extended[i + 1]
            plt.arrow(np_points[start_idx, 0], np_points[start_idx, 1],
                        np_points[end_idx, 0] - np_points[start_idx, 0],
                        np_points[end_idx, 1] - np_points[start_idx, 1],
                        head_width=arrow_head_width, head_length=arrow_head_length, 
                        fc='blue', ec='blue', alpha=0.7, linewidth=1.5)
        
        plt.title(f'Ground Truth Solution - Cost: {gt_cost:.2f}', fontsize=12)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 绘制扩散模型求解结果
        plt.subplot(1, 2, 2)
        plt.scatter(np_points[:, 0], np_points[:, 1], c='red', s=50, zorder=5, alpha=0.6)
        for i, (x, y) in enumerate(np_points):
            plt.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points', 
                        fontsize=8, ha='left')
        
        # 绘制扩散模型求解路径 - 使用较小的箭头
        for i in range(len(best_tour) - 1):
            start_idx = best_tour[i]
            end_idx = best_tour[i + 1]
            plt.arrow(np_points[start_idx, 0], np_points[start_idx, 1],
                        np_points[end_idx, 0] - np_points[start_idx, 0],
                        np_points[end_idx, 1] - np_points[start_idx, 1],
                        head_width=arrow_head_width, head_length=arrow_head_length, 
                        fc='green', ec='green', alpha=0.7, linewidth=1.5)
        
        plt.title(f'Diffusion Model Solution - Cost: {best_real_cost:.2f}', fontsize=12)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(f'tsp_comparison_sample_{idx+1}.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"\nComparison plot saved as 'tsp_comparison_sample_{idx+1}.png'")
        plt.close()
        
    except ImportError:
        print("\nNote: matplotlib not installed, skipping visualization")