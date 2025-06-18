"""
这个文档将被用于导入训练好的difussion模型，用于生成节点之间的adjacency matrix
PYTHONPATH=~/miniconda3/envs/difusco_cp/bin/python
"""

# 将特定文件夹导入到python环境中
import sys
import numpy as np
import torch
import torch.utils.data

import os
from argparse import ArgumentParser

sys.path.append('/root/code/diffusco_o/difusco_o/DIFUSCO/difusco') 

from utils.diffusion_schedulers import InferenceSchedule
from pl_tsp_model import TSPModel
from utils.draw_utils import visualize_tsp_solutions


class TSPModel_v2(TSPModel):
    def __init__(self, param_args=None, pro_idx=0):
        super(TSPModel_v2, self).__init__(param_args=param_args)
        self.args = param_args

    def generate_adj(self, points, split='test'):
        """
        生成TSP问题的邻接矩阵
        Args:
            points: numpy array of shape (batch_size, num_nodes, 2) - 节点坐标
        Returns:
            adj_mat: numpy array of shape (batch_size, num_nodes, num_nodes) - 邻接矩阵
        """
        device = next(self.parameters()).device
        
        # 确保points是正确的格式
        if isinstance(points, np.ndarray):
            np_points = points
        else:
            np_points = points.cpu().numpy()
            
        # 确保是3维数组 (batch_size, num_nodes, 2)
        if len(np_points.shape) == 2:
            np_points = np_points[np.newaxis, :, :]
            
        batch_size, num_nodes, _ = np_points.shape

        for _ in range(self.args.sequential_sampling):
            # 初始化随机噪声矩阵
            xt = torch.randn(batch_size, num_nodes, num_nodes, device=device)
            
            if self.args.parallel_sampling > 1:
                xt = xt.repeat(self.args.parallel_sampling, 1, 1)
                xt = torch.randn_like(xt)

            if self.diffusion_type == 'gaussian':
                xt.requires_grad = True
            else:
                xt = (xt > 0).long()

            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=self.args.inference_schedule,
                T=self.diffusion.T, 
                inference_T=steps
            )

            # 扩散迭代
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)

                # t1 = t1.repeat(batch_size)
                # t2 = t2.repeat(batch_size)

                points_tensor = torch.from_numpy(np_points).float().to(device)

                if self.diffusion_type == 'gaussian':
                    xt = self.gaussian_denoise_step(points_tensor, xt, t1, device, None, target_t=t2)
                else:
                    xt = self.categorical_denoise_step(points_tensor, xt, t1, device, None, target_t=t2)

            if self.diffusion_type == 'gaussian':
                adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
            else:
                adj_mat = xt.float().cpu().detach().numpy() + 1e-6

            if self.args.save_numpy_heatmap:
                self.run_save_numpy_heatmap(adj_mat, np_points, split)
                
        return adj_mat


def greedy_tsp_solver(adj_matrix, start_node=0):
    """
    基于邻接矩阵的贪婪TSP求解器
    Args:
        adj_matrix: numpy array of shape (num_nodes, num_nodes) - 邻接矩阵
        start_node: int - 起始节点索引
    Returns:
        tour: list - TSP路径
        total_cost: float - 总成本
    """
    num_nodes = adj_matrix.shape[0]
    visited = set([start_node])
    tour = [start_node]
    current_node = start_node
    total_cost = 0.0
    
    # 贪婪选择下一个节点
    for _ in range(num_nodes - 1):
        best_next_node = None
        best_cost = float('-inf')  # 修复：初始化为负无穷，因为我们要找最大权重
        
        # 找到未访问节点中权重最大的连接
        for next_node in range(num_nodes):
            if next_node not in visited:
                # 使用邻接矩阵中的值作为选择概率/权重
                edge_weight = adj_matrix[current_node, next_node]
                if edge_weight > best_cost:
                    best_cost = edge_weight
                    best_next_node = next_node
        
        # 添加到路径中
        if best_next_node is not None:
            visited.add(best_next_node)
            tour.append(best_next_node)
            total_cost += best_cost
            current_node = best_next_node
    
    # 回到起始节点
    tour.append(start_node)
    total_cost += adj_matrix[current_node, start_node]
    
    return tour, total_cost


def calculate_euclidean_distance(points):
    """
    计算节点间的欧几里得距离矩阵
    Args:
        points: numpy array of shape (num_nodes, 2)
    Returns:
        distance_matrix: numpy array of shape (num_nodes, num_nodes)
    """
    num_nodes = points.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i, j] = np.sqrt(
                    (points[i, 0] - points[j, 0])**2 + 
                    (points[i, 1] - points[j, 1])**2
                )
    
    return distance_matrix


def calculate_tour_cost(tour, distance_matrix):
    """
    计算路径的实际成本
    Args:
        tour: list - 节点访问顺序
        distance_matrix: numpy array - 真实距离矩阵
    Returns:
        total_cost: float - 总成本
    """
    total_cost = 0.0
    for i in range(len(tour) - 1):
        total_cost += distance_matrix[tour[i], tour[i + 1]]
    return total_cost


def arg_parser():
    parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')
    parser.add_argument('--task', type=str, default='tsp')
    parser.add_argument('--storage_path', type=str, default="./")
    parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
    parser.add_argument('--training_split_label_dir', type=str, default=None,
                        help="Directory containing labels for training split (used for MIS).")
    parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
    parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
    parser.add_argument('--validation_examples', type=int, default=64)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', type=str, default='constant')

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_activation_checkpoint', action='store_true')

    parser.add_argument('--diffusion_type', type=str, default='categorical')  # categorical or gaussian
    parser.add_argument('--diffusion_schedule', type=str, default='cosine') # linear or cosine
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_diffusion_steps', type=int, default=5)  # 减少推理步数以加快速度  5步保证显存够用
    parser.add_argument('--inference_schedule', type=str, default='linear')   # linear or cosine
    parser.add_argument('--inference_trick', type=str, default="ddim")
    parser.add_argument('--sequential_sampling', type=int, default=1)
    parser.add_argument('--parallel_sampling', type=int, default=1)

    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--sparse_factor', type=int, default=-1)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--two_opt_iterations', type=int, default=1000)
    parser.add_argument('--save_numpy_heatmap', default=False)

    parser.add_argument('--project_name', type=str, default='tsp_diffusion')
    parser.add_argument('--ckpt_path', type=str, default='tb_logs/tsp_diffusion/version_0/checkpoints/last.ckpt')
    parser.add_argument('--resume_weight_only',  default=True)

    parser.add_argument('--do_train', default=False)
    parser.add_argument('--do_test', default=False)
    parser.add_argument('--do_valid_only', default=False)
    
    args = parser.parse_args()
    return args


# 全局加载模型（避免重复加载）
args = arg_parser()
tspmodel = None

def load_model():
    global tspmodel
    if tspmodel is None:
        if args.resume_weight_only:
            tspmodel = TSPModel_v2.load_from_checkpoint(args.ckpt_path, param_args=args)
            # 强制迁移所有模型参数到CUDA
            tspmodel = tspmodel.cuda()
            tspmodel.eval()
            # 检查设备
            device = next(tspmodel.parameters()).device
            print(f'成功加载训练好的模型: {args.ckpt_path}')
            print(f'模型设备: {device}')
        else:
            tspmodel = TSPModel_v2(param_args=args).cuda()
            tspmodel.eval()
            print('使用未训练的模型')
    return tspmodel


if __name__ == '__main__':
    # 加载模型
    model = load_model()
    model.eval()
    
    # 获取模型设备
    model_device = next(model.parameters()).device
    print(f"模型设备: {model_device}")
    
    # 获取测试数据加载器
    test_dataloader = model.test_dataloader()
    print(f"测试数据集大小: {len(model.test_dataset)}")
    
    # 随机采样一批问题进行测试
    import random
    batch_indices = random.sample(range(len(test_dataloader)), min(1, len(test_dataloader)))
    print(f"随机选择 {len(batch_indices)} 个测试样例进行求解")
    
    for idx, batch_idx in enumerate(batch_indices):
        print(f"\n=== 测试样例 {idx + 1}/{len(batch_indices)} (数据集索引: {batch_idx}) ===")
        
        # 获取对应的batch数据
        batch = list(test_dataloader)[batch_idx]
        
        # 解析batch数据 (仿照test_step的开始部分)
        edge_index = None
        np_edge_index = None
        device = model_device  # 使用模型设备而不是数据设备
        
        if not model.sparse:
            real_batch_idx, points, adj_matrix, gt_tour = batch
            np_points = points.cpu().numpy()[0]
            np_gt_tour = gt_tour.cpu().numpy()[0]
            # 强制迁移到模型设备
            points = points.to(device)
            adj_matrix = adj_matrix.to(device)
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
            # 强制迁移到模型设备
            points = points.to(device)
            adj_matrix = adj_matrix.to(device)
            if edge_index is not None:
                edge_index = edge_index.to(device)
        
        print(f"节点数量: {len(np_points)}")
        print(f"真实最优路径: {np_gt_tour}")
        
        # 进行扩散求解 (仿照test_step的扩散部分)
        stacked_tours = []
        
        with torch.no_grad():
            for _ in range(model.args.sequential_sampling):
                # 初始化噪声 - 确保在正确的设备上
                xt = torch.randn_like(adj_matrix.float()).to(device)
                
                # 确保points在正确的设备上
                if not model.sparse:
                    points = points.to(device)
                else:
                    points = points.to(device)
                    if edge_index is not None:
                        edge_index = edge_index.to(device)
                
                if model.args.parallel_sampling > 1:
                    if not model.sparse:
                        xt = xt.repeat(model.args.parallel_sampling, 1, 1)
                        points = points.repeat(model.args.parallel_sampling, 1, 1)
                    else:
                        xt = xt.repeat(model.args.parallel_sampling, 1)
                        points = points.repeat(model.args.parallel_sampling, 1)
                        edge_index = model.duplicate_edge_index(edge_index, np_points.shape[0], device)
                    xt = torch.randn_like(xt).to(device)

                if model.diffusion_type == 'gaussian':
                    xt.requires_grad = True
                else:
                    xt = (xt > 0).long()

                if model.sparse:
                    xt = xt.reshape(-1)

                # 扩散时间调度
                steps = model.args.inference_diffusion_steps
                time_schedule = InferenceSchedule(
                    inference_schedule=model.args.inference_schedule,
                    T=model.diffusion.T, 
                    inference_T=steps
                )

                # 扩散迭代 (仿照test_step的扩散循环)
                for i in range(steps):
                    t1, t2 = time_schedule(i)
                    t1 = np.array([t1]).astype(int)
                    t2 = np.array([t2]).astype(int)

                    if model.diffusion_type == 'gaussian':
                        xt = model.gaussian_denoise_step(
                            points, xt, t1, device, edge_index, target_t=t2)
                    else:
                        xt = model.categorical_denoise_step(
                            points, xt, t1, device, edge_index, target_t=t2)

                # 获取最终的邻接矩阵
                if model.diffusion_type == 'gaussian':
                    adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
                else:
                    adj_mat = xt.float().cpu().detach().numpy() + 1e-6

                # 处理不同采样情况
                if len(adj_mat.shape) == 3:
                    # 批次数据，取第一个实例（因为测试时batch_size=1）
                    adj_mat = adj_mat[0]
                elif model.args.parallel_sampling > 1:
                    # 处理并行采样 - 评估所有采样结果，选择最优的
                    adj_mat = adj_mat.reshape(model.args.parallel_sampling, 
                                            len(np_points), len(np_points))
                    
                    # 为每个采样结果找到最优解
                    sampling_results = []
                    distance_matrix = calculate_euclidean_distance(np_points)
                    
                    for sample_idx in range(model.args.parallel_sampling):
                        sample_adj_mat = adj_mat[sample_idx]
                        sample_best_cost = float('inf')
                        sample_best_tour = None
                        
                        # 对当前采样结果尝试多个起始点
                        for start_node in range(min(len(np_points), 5)):
                            tour, _ = greedy_tsp_solver(sample_adj_mat, start_node)
                            real_cost = calculate_tour_cost(tour, distance_matrix)
                            
                            if real_cost < sample_best_cost:
                                sample_best_cost = real_cost
                                sample_best_tour = tour
                        
                        sampling_results.append((sample_best_cost, sample_best_tour, sample_adj_mat))
                    
                    # 选择最优的采样结果
                    best_sampling_result = min(sampling_results, key=lambda x: x[0])
                    adj_mat = best_sampling_result[2]  # 使用最优采样的邻接矩阵
                    print(f"并行采样结果: {len(sampling_results)} 个采样，最优成本: {best_sampling_result[0]:.4f}")
                
                # 将当前采样的adj_mat添加到列表中，用于后续可能的sequential_sampling处理
                stacked_tours.append(adj_mat)
        
        print(f"生成的邻接矩阵形状: {adj_mat.shape}")
        print(f"邻接矩阵样例 (前5x5):")
        print(adj_mat[:5, :5])
        
        # 对所有采样结果进行评估，选择最优的
        print(f"开始使用贪婪求解器求解...")
        
        # 计算真实的欧几里得距离矩阵
        distance_matrix = calculate_euclidean_distance(np_points)
        
        # 对所有sequential_sampling结果进行评估
        all_results = []
        for seq_idx, seq_adj_mat in enumerate(stacked_tours):
            seq_best_cost = float('inf')
            seq_best_tour = None
            
            # 尝试多个起始点
            for start_node in range(min(len(np_points), 5)):
                tour, adj_cost = greedy_tsp_solver(seq_adj_mat, start_node)
                real_cost = calculate_tour_cost(tour, distance_matrix)
                
                if real_cost < seq_best_cost:
                    seq_best_cost = real_cost
                    seq_best_tour = tour
            
            all_results.append((seq_best_cost, seq_best_tour))
        
        # 选择最优结果
        if all_results:
            best_result = min(all_results, key=lambda x: x[0])
            best_real_cost, best_tour = best_result
            sampling_info = f"{len(all_results)} 个顺序采样"
            if model.args.parallel_sampling > 1:
                sampling_info += f" × {model.args.parallel_sampling} 个并行采样"
            print(f"采样结果: {sampling_info}，最优成本: {best_real_cost:.4f}")
        else:
            # 降级处理，如果没有结果就用最后一个adj_mat
            best_tour = None
            best_real_cost = float('inf')
            for start_node in range(min(len(np_points), 5)):
                tour, adj_cost = greedy_tsp_solver(adj_mat, start_node)
                real_cost = calculate_tour_cost(tour, distance_matrix)
                if real_cost < best_real_cost:
                    best_real_cost = real_cost
                    best_tour = tour
        
        # 计算真实最优路径的成本
        if 'distance_matrix' not in locals():
            distance_matrix = calculate_euclidean_distance(np_points)
        gt_cost = calculate_tour_cost(list(np_gt_tour) + [np_gt_tour[0]], distance_matrix)
        
        print(f"=== 求解结果 ===")
        print(f"真实最优路径成本: {gt_cost:.4f}")
        print(f"扩散模型求解路径: {best_tour}")
        print(f"扩散模型求解成本: {best_real_cost:.4f}")
        print(f"相对差距: {((best_real_cost - gt_cost) / gt_cost * 100):.2f}%")
        
        # 可视化结果（仅为第一个样例）
        if idx == 0: 
          # 调用可视化函数
          visualize_tsp_solutions(np_points, np_gt_tour, gt_cost, best_tour, best_real_cost, idx)
    
    print(f"\n=== 测试完成 ===")