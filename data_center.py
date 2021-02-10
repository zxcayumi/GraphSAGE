import os.path as path
import numpy as np
from numpy.core.defchararray import replace
import loader
from enum import Enum

class set_type(Enum):
    trains = 0
    vals = 1
    tests = 2

class data_center:
    def __init__(self,data_name):
        self.data_name = data_name
        self.loader = getattr(loader, data_name + '_loader') #反射类类型
        self.data = self.loader(self.data_name) #实例化类

        self.rand_data_index = np.random.permutation(self.data.count)
        self.tests_indexs = self.rand_data_index[ : self.data.count//3]
        self.vals_indexs = self.rand_data_index[self.data.count//3 : self.data.count//3 + self.data.count//6]
        self.trains_indexs = self.rand_data_index[self.data.count//3 + self.data.count//6 : ]
    
    @property
    def trans(self):
        'training set'
        return self.data.features[self.trains_indexs]

    @property
    def vals(self):
        'validating set'
        return self.data.features[self.vals_indexs]

    @property
    def tests(self):
        'tesing set'
        return self.data.features[self.tests_indexs]

    def sampling(self, src_nodes_batch, neighbors_num, from_set = set_type.trains):
        ''' 
        src_nodes_batch: number of source nodes 
        neighbors_num: list type, e.g. [10,10] (sampling second order neighbor, every layer has 10 neighbors)
        from_set: which dataset does the data come from
        '''
        data_indexs = getattr(self,from_set.name + '_indexs')

        result = []
        src_nodes_index = np.random.choice(data_indexs, size = src_nodes_batch)
        result.append(src_nodes_index)
        
        for i,num in enumerate(neighbors_num): #sampling neighbors by order
            samp_neighbors = []
            for node_index in result[i]: #sampling the ith layer's neighbors
                all_neighbors = self.data.adj[node_index]
                samp_neighbors.append(np.random.choice(all_neighbors, num))

            result.append(np.asarray(samp_neighbors).flatten()) #append the (i+1)layer's nodes
        
        return src_nodes_index, self._to_features(result)

    def _to_features(self, sampling_indexs):
        'convert node index to features'
        result = []
        for nodes in sampling_indexs:
            result.append(self.data.features[nodes])
        
        return result

if __name__ == '__main__':
    #dc = data_center('cora')
    # print(dc.data.features,dc.data.features.shape)
    # print(type(dc.tests_indexs))
    # print(dc.trans.shape)
    # print(len(dc.trains_indexs) + len(dc.tests_indexs) + len(dc.vals_indexs))
    # print(dc.data.features[[1,4,7,332,983]])
    # print(dc.test())
    # samps = dc.sampling(2,[3,5])
    # for item in samps:
    #     print(item.shape)
    # print(samps)
    
    import torch

    result = torch.Tensor([[0.212,.53114,-0.125]])
    print(result.max(1))

