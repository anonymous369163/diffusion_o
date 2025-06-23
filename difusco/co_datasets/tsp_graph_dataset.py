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
  def __init__(self, data_file='../../data/tsp/vrp_dataset.pkl', sparse_factor=-1, data_size=None, start_idx=0):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    with open(data_file, 'rb') as f:
      self.file_lines = pickle.load(f)

    if data_size is not None:
      self.data_size = data_size
      self.file_lines['node_xy'] = self.file_lines['node_xy'][start_idx:start_idx+data_size]
      self.file_lines['route'] = self.file_lines['route'][start_idx:start_idx+data_size]


  def __len__(self):
    return len(self.file_lines["node_xy"])

  def get_example(self, idx):
    # Select sample from self.file_lines
    tour = self.file_lines["route"][idx]
    points = self.file_lines["node_xy"][idx]
    points = points[:, :2]
    # if  idx < 1280:
    #   zeros = np.zeros((points.shape[0], 3))   # d, e, l 新引入的三个属性，分别表示节点需求，time window的early time 和 late time 
    #   points = np.concatenate([points, zeros], axis=1)
    # else:
    #   ones = np.ones((points.shape[0], 2))   # d, l 新引入的两个属性，分别表示节点需求和time window的late time 
    #   points = np.concatenate([points, ones], axis=1)
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
      # 对于稀疏图,抛出NotImplementedError异常
      raise NotImplementedError("稀疏图的处理尚未实现")