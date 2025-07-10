"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch

from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


class TSPGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])
    tour -= 1

    return points, tour

  def __getitem__(self, idx):
    points, tour = self.get_example(idx)
    if self.sparse_factor <= 0:
      # Return a densely connected graph
      adj_matrix = np.zeros((points.shape[0], points.shape[0]))
      for i in range(tour.shape[0] - 1):
        adj_matrix[tour[i], tour[i + 1]] = 1
      # return points, adj_matrix, tour
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          torch.from_numpy(tour).long(),
      )
    else:
      # Return a sparse graph where each node is connected to its k nearest neighbors
      # k = self.sparse_factor
      sparse_factor = self.sparse_factor
      kdt = KDTree(points, leaf_size=30, metric='euclidean')
      dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

      edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

      edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

      tour_edges = np.zeros(points.shape[0], dtype=np.int64)
      tour_edges[tour[:-1]] = tour[1:]
      tour_edges = torch.from_numpy(tour_edges)
      tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
      graph_data = GraphData(x=torch.from_numpy(points).float(),
                             edge_index=edge_index,
                             edge_attr=tour_edges)

      point_indicator = np.array([points.shape[0]], dtype=np.int64)
      edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          graph_data,
          torch.from_numpy(point_indicator).long(),
          torch.from_numpy(edge_indicator).long(),
          torch.from_numpy(tour).long(),
      )

import pickle

class VRPGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    with open(data_file, 'rb') as f:
      self.file_lines = pickle.load(f)
    
    # pre-process the data
    # 列车的容量初始容量都是1，解码过程中列车的实际容量，当前时间，所行驶的长度和路径是否需要返回到起点，这些都是动态属性，需要解码过程中动态计算，便于处理约束
    self.file_lines["depot_xy"] = np.concatenate(self.file_lines["depot_xy"], axis=0)
    self.file_lines["node_xy"] = np.concatenate(self.file_lines["node_xy"], axis=0)
    self.file_lines["node_demand"] = np.concatenate(self.file_lines["node_demand"], axis=0)
    self.file_lines["node_earlyTW"] = np.concatenate(self.file_lines["node_earlyTW"], axis=0)
    self.file_lines["node_lateTW"] = np.concatenate(self.file_lines["node_lateTW"], axis=0)
    self.file_lines["route_open"] = np.concatenate(self.file_lines["route_open"], axis=0)
    self.file_lines["length"] = np.concatenate(self.file_lines["length"], axis=0)
    self.file_lines["tours"] = np.concatenate(self.file_lines["tours"], axis=0)

    print(f'Loaded "{data_file}" with {len(self.file_lines["depot_xy"])} lines')

  def __len__(self):
    return len(self.file_lines["depot_xy"])

  def get_example(self, idx):
    # Select sample from self.file_lines
    depot_xy = self.file_lines["depot_xy"][idx]
    node_xy = self.file_lines["node_xy"][idx]
    node_demand = self.file_lines["node_demand"][idx]
    node_earlyTW = self.file_lines["node_earlyTW"][idx]
    node_lateTW = self.file_lines["node_lateTW"][idx]
    route_open = self.file_lines["route_open"][idx]
    length = self.file_lines["length"][idx]
    solutions = self.file_lines["tours"][idx]

    # 每个维度都加上一些额外的属性
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

  def __getitem__(self, idx):
    points, tour = self.get_example(idx)
    if self.sparse_factor <= 0:
      # Return a densely connected graph
      adj_matrix = np.zeros((points.shape[0], points.shape[0]))
      for i in range(tour.shape[0] - 1):
        adj_matrix[tour[i], tour[i + 1]] = 1
      # return points, adj_matrix, tour
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          torch.from_numpy(tour).long(),
      )
    else:   
      # Return a sparse graph where each node is connected to its k nearest neighbors
      # k = self.sparse_factor
      raise NotImplementedError("Sparse graph is not supported for VRP")