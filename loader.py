from typing import DefaultDict
import numpy as np
from torch._C import dtype

class loader(object):
    def __init__(self, data_name):
        self.data_name = data_name
        self.features = []
        self.labels = []
        self.node_map = {}
        self.label_map = {}
        self.adj = DefaultDict(list)
        self.load() #加载数据

    def load(self):
        pass

    @property
    def count(self):
        return len(self.labels)

class cora_loader(loader):
    'cora数据集加载'
    def load(self):
        print('init:数据加载...')
        path = 'data/' + self.data_name + '/'
        with open(path + 'cora.content') as f:
            print('init:加载特征...')
            for i,line in enumerate(f):
                content = line.strip().split()
                #节点特征
                self.features.append([float(item) for item in content[1:-1]])

                self.node_map[content[0]] = i #构造“节点-->编号”映射字典
                if content[-1] not in self.label_map: #构造“label-->编号”映射字典
                    self.label_map[content[-1]] = len(self.label_map)
                
                #label-->编号
                label_index = self.label_map[content[-1]]
                self.labels.append(label_index)

            self.features = np.asarray(self.features, dtype=np.float32)
            self.labels = np.asarray(self.labels, dtype=np.int64)
                
        with open(path + 'cora.cites') as f:
            print('init:构造邻居...')
            for i,line in enumerate(f):
                content = line.strip().split()
                from_index = self.node_map[content[0]]
                to_index = self.node_map[content[1]]
                self.adj[from_index].append(to_index) #构造邻接信息字典key:节点编号, vale:邻接点set
                self.adj[to_index].append(from_index) #undirected graph

class citeseer_loader(loader):
    'citeseer数据集加载'
    def load(self):
        print('init:数据加载...')
        path = 'data/' + self.data_name + '/'
        with open(path + 'citeseer.content') as f:
            print('init:加载特征...')
            for i,line in enumerate(f):
                content = line.strip().split()
                #节点特征
                self.features.append([float(item) for item in content[1:-1]])

                self.node_map[content[0]] = i #构造“节点-->编号”映射字典
                if content[-1] not in self.label_map: #构造“label-->编号”映射字典
                    self.label_map[content[-1]] = len(self.label_map)
                
                #label-->编号
                label_index = self.label_map[content[-1]]
                self.labels.append(label_index)

            self.features = np.asarray(self.features, dtype=np.float32)
            self.labels = np.asarray(self.labels, dtype=np.int64)
                
        with open(path + 'citeseer.cites') as f:
            print('init:构造邻居...')
            for i,line in enumerate(f):
                content = line.strip().split()
                from_index = self.node_map[content[0]]
                to_index = self.node_map[content[1]]
                self.adj[from_index].append(to_index) #构造邻接信息字典key:节点编号, vale:邻接点set
                self.adj[to_index].append(from_index) #undirected graph

class pubmed_loader(loader):
    'pubmed数据集加载'
    def load(self):
        pass