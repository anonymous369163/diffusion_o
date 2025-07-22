# DIFUSCO Package
# 扩散模型用于组合优化问题

__version__ = "1.0.0"
__author__ = "DIFUSCO Team"

# 由于模块间的复杂依赖，我们提供一个函数来安全地导入核心函数
def get_core_functions():
    """
    安全地导入核心函数，处理依赖问题
    返回一个包含核心函数的字典
    """
    import sys
    import os
    
    # 添加项目根目录到 Python 路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        # 尝试导入完整模块
        from .pl_tsp_model import (
            calculate_euclidean_distance_batch,
            calculate_tour_cost_batch,
            calculate_tour_cost_batch_pomo,
            greedy_solver_batch_pomo,
            enhance_adjacency_matrix
        )
        
        return {
            'calculate_euclidean_distance_batch': calculate_euclidean_distance_batch,
            'calculate_tour_cost_batch': calculate_tour_cost_batch,
            'calculate_tour_cost_batch_pomo': calculate_tour_cost_batch_pomo,
            'greedy_solver_batch_pomo': greedy_solver_batch_pomo,
            'enhance_adjacency_matrix': enhance_adjacency_matrix,
            'import_success': True,
            'import_method': 'full_module'
        }
    except ImportError as e:
        print(f"完整模块导入失败: {e}")
        return {
            'import_success': False,
            'error': str(e),
            'import_method': 'failed'
        }

# 导出的内容
__all__ = ['get_core_functions']
