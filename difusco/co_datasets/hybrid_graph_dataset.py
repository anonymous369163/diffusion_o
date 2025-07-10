"""混合VRP变体数据集类"""

import numpy as np
import torch
import pickle
import os
import random
from typing import List, Dict, Tuple, Any


class HybridGraphDataset(torch.utils.data.Dataset):
    """
    混合VRP变体数据集类，支持多个VRP变体的合并训练
    确保每个batch内的问题类型一致，并在__getitem__中返回问题类型
    """
    
    def __init__(self, data_configs: List[Dict[str, str]], sparse_factor: int = -1, batch_size: int = 32):
        """
        初始化混合数据集
        
        Args:
            data_configs: 数据配置列表，每个配置包含：
                - 'data_file': 数据文件路径
                - 'problem_type': 问题类型（如 'CVRP', 'VRPTW', 'OVRP' 等）
            sparse_factor: 稀疏因子，-1表示密集图
            batch_size: 批次大小，用于确保每个batch内问题类型一致
        """
        self.data_configs = data_configs
        self.sparse_factor = sparse_factor
        self.batch_size = batch_size
        
        # 加载所有数据文件
        self.datasets = []
        self.problem_types = []
        self.data_indices = []  # 记录每个样本对应的数据集索引
        
        total_samples = 0
        
        for config in data_configs:
            data_file = config['data_file']
            problem_type = config['problem_type']
            
            # 加载数据
            if problem_type == "TSP":
                # TSP数据格式
                file_lines = open(data_file).read().splitlines()
                dataset_size = len(file_lines)
                dataset_data = file_lines
            else:
                # VRP数据格式
                with open(data_file, 'rb') as f:
                    file_lines = pickle.load(f)
                
                # 预处理VRP数据
                processed_data = {
                    "depot_xy": np.concatenate(file_lines["depot_xy"], axis=0),
                    "node_xy": np.concatenate(file_lines["node_xy"], axis=0),
                    "node_demand": np.concatenate(file_lines["node_demand"], axis=0),
                    "node_earlyTW": np.concatenate(file_lines["node_earlyTW"], axis=0),
                    "node_lateTW": np.concatenate(file_lines["node_lateTW"], axis=0),
                    "route_open": np.concatenate(file_lines["route_open"], axis=0),
                    "length": np.concatenate(file_lines["length"], axis=0),
                    "tours": np.concatenate(file_lines["tours"], axis=0)
                }
                
                dataset_size = len(processed_data["depot_xy"])
                dataset_data = processed_data
            
            self.datasets.append(dataset_data)
            self.problem_types.append(problem_type)
            
            # 记录每个样本的数据集索引
            for i in range(dataset_size):
                self.data_indices.append(len(self.datasets) - 1)
            
            total_samples += dataset_size
            
            print(f'加载 "{data_file}" ({problem_type})，包含 {dataset_size} 个样本')
        
        print(f'总共加载 {total_samples} 个样本，涵盖 {len(self.data_configs)} 种问题类型')
        
        # 创建批次索引映射，确保每个batch内问题类型一致
        self._create_batch_mappings()
    
    def _create_batch_mappings(self):
        """创建批次索引映射，确保每个batch内的问题类型一致"""
        # 按问题类型分组样本索引
        type_to_indices = {}
        for idx, dataset_idx in enumerate(self.data_indices):
            problem_type = self.problem_types[dataset_idx]
            if problem_type not in type_to_indices:
                type_to_indices[problem_type] = []
            type_to_indices[problem_type].append(idx)
        
        # 为每种问题类型创建批次
        self.batch_mappings = []
        self.batch_types = []
        
        for problem_type, indices in type_to_indices.items():
            # 打乱该类型的样本索引
            random.shuffle(indices)
            
            # 创建批次
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                self.batch_mappings.append(batch_indices)
                self.batch_types.append(problem_type)
        
        # 打乱批次顺序
        combined = list(zip(self.batch_mappings, self.batch_types))
        random.shuffle(combined)
        self.batch_mappings, self.batch_types = zip(*combined)
        
        print(f'创建了 {len(self.batch_mappings)} 个批次，确保每个批次内问题类型一致')
    
    def __len__(self):
        """返回数据集总长度"""
        return len(self.data_indices)
    
    def get_example(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取单个样本的数据"""
        dataset_idx = self.data_indices[idx]
        dataset = self.datasets[dataset_idx]
        problem_type = self.problem_types[dataset_idx]
        
        if problem_type == "TSP":
            # TSP数据处理
            # 计算在该数据集中的相对索引
            relative_idx = self._get_relative_idx(idx, dataset_idx)
            line = dataset[relative_idx]
            line = line.strip()
            
            # 提取坐标点
            points = line.split(' output ')[0]
            points = points.split(' ')
            points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
            
            # 提取路径
            tour = line.split(' output ')[1]
            tour = tour.split(' ')
            tour = np.array([int(t) for t in tour])
            tour -= 1
            
            return points, tour
        else:
            # VRP数据处理
            relative_idx = self._get_relative_idx(idx, dataset_idx)
            
            depot_xy = dataset["depot_xy"][relative_idx]
            node_xy = dataset["node_xy"][relative_idx]
            node_demand = dataset["node_demand"][relative_idx]
            node_earlyTW = dataset["node_earlyTW"][relative_idx]
            node_lateTW = dataset["node_lateTW"][relative_idx]
            route_open = dataset["route_open"][relative_idx]
            length = dataset["length"][relative_idx]
            solutions = dataset["tours"][relative_idx]
            
            # 构建特征矩阵
            depot_xy = np.concatenate([depot_xy, np.zeros((1, 5))], axis=1)
            points = np.concatenate([node_xy,
                                     node_demand.reshape(-1, 1),
                                     node_earlyTW.reshape(-1, 1),
                                     node_lateTW.reshape(-1, 1),
                                     route_open.reshape(-1, 1),
                                     length.reshape(-1, 1)], axis=1)
            points = np.concatenate([depot_xy, points], axis=0)
            tour = solutions
            
            return points, tour
    
    def _get_relative_idx(self, global_idx: int, dataset_idx: int) -> int:
        """获取在指定数据集中的相对索引"""
        # 计算在该数据集之前的所有样本数量
        samples_before = 0
        for i in range(dataset_idx):
            if self.problem_types[i] == "TSP":
                samples_before += len(self.datasets[i])
            else:
                samples_before += len(self.datasets[i]["depot_xy"])
        
        return global_idx - samples_before
    
    def _get_dataset_start_idx(self, dataset_idx: int) -> int:
        """获取指定数据集的起始索引"""
        start_idx = 0
        for i in range(dataset_idx):
            if self.problem_types[i] == "TSP":
                start_idx += len(self.datasets[i])
            else:
                start_idx += len(self.datasets[i]["depot_xy"])
        return start_idx
    
    def get_problem_type(self, idx: int) -> str:
        """获取指定索引样本的问题类型"""
        dataset_idx = self.data_indices[idx]
        return self.problem_types[dataset_idx]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        获取单个样本
        
        Returns:
            Tuple containing:
            - sample_idx: 样本索引
            - points: 节点特征
            - adj_matrix: 邻接矩阵
            - tour: 路径
            - problem_type: 问题类型
        """
        points, tour = self.get_example(idx)
        problem_type = self.get_problem_type(idx)
        
        if self.sparse_factor <= 0:
            # 密集图
            adj_matrix = np.zeros((points.shape[0], points.shape[0]))
            for i in range(tour.shape[0] - 1):
                adj_matrix[tour[i], tour[i + 1]] = 1
            
            return (
                torch.LongTensor(np.array([idx], dtype=np.int64)),
                torch.from_numpy(points).float(),
                torch.from_numpy(adj_matrix).float(),
                torch.from_numpy(tour).long(),
                problem_type  # 新增：返回问题类型
            )
        else:
            # 稀疏图
            raise NotImplementedError("混合数据集暂不支持稀疏图")
    
    def get_batch_by_type(self, batch_idx: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]]:
        """
        根据批次索引获取同类型的批次数据
        
        Args:
            batch_idx: 批次索引
            
        Returns:
            List of samples for the batch
        """
        if batch_idx >= len(self.batch_mappings):
            raise IndexError(f"批次索引 {batch_idx} 超出范围")
        
        sample_indices = self.batch_mappings[batch_idx]
        batch_type = self.batch_types[batch_idx]
        
        batch_data = []
        for sample_idx in sample_indices:
            sample_data = self.__getitem__(sample_idx)
            batch_data.append(sample_data)
        
        return batch_data, batch_type
    
    def get_num_batches(self) -> int:
        """获取批次总数"""
        return len(self.batch_mappings)
    
    def shuffle_batches(self):
        """重新打乱批次顺序"""
        self._create_batch_mappings() 