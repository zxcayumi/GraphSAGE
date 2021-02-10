from numpy.core.defchararray import mod
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch.nn.modules import activation
import util

class Classification(nn.Module):
    def __init__(self, emb_dim, class_dim):
        super(Classification, self).__init__()
        self.emb_dim = emb_dim
        self.class_size = class_dim
        self.fcl = nn.Linear(emb_dim,class_dim)

        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self,embedings):
        embedings = self.fcl(embedings)
        #logists = torch.log_softmax(embedings,dim=1)
        return embedings

class SageGCN(nn.Module):
    def __init__(self,input_dim, hidden_dim, \
        aggr_type_neighbor='mean', aggr_type='sum'):
        super(SageGCN,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_type_neighbor = aggr_type_neighbor
        self.aggr_type = aggr_type
        self.weights1 = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.weights2 = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.activation = func.relu

        self.init_params()
    
    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
    
    def forward(self,cur_features,neighbor_features):
        '''
        cur_features: the features of current layer nodes
        neighbor_features: the features of current layer nodes' neighbors
        return_value: the features of aggreation
        '''
        #aggregate neighbors and reduce dimension
        if self.aggr_type_neighbor == 'mean': aggregated_features_neighbor = neighbor_features.mean(dim=1)
        elif self.aggr_type_neighbor == 'sum': aggregated_features_neighbor = neighbor_features.sum(dim=1)
        else: aggregated_features_neighbor = neighbor_features.mean(dim=1)
        aggregated_features_neighbor = torch.matmul(aggregated_features_neighbor, self.weights1)

        #aggreate current nodes and its' neighbors
        aggregated_features_cur = torch.matmul(cur_features, self.weights2) #当前层节点和其邻居节点的变换用相同的还是不同的变换矩阵
        if self.aggr_type == 'sum': aggregated_features_cur += aggregated_features_neighbor
        elif self.aggr_type == 'concat': aggregated_features_cur = torch.cat([aggregated_features_cur, aggregated_features_neighbor], dim=1)
        else: aggregated_features_cur += aggregated_features_neighbor

        return self.activation(aggregated_features_cur)

class GraphSage(nn.Module):
    def __init__(self, input_dim, output_dim, orders):
        '''
        nodes_feature: batch nodes' feafure with their n order neighbors, e.g. [array(), array(), array()]
        '''
        self._orders = orders
        super(GraphSage,self).__init__()
        self.gcnList = nn.ModuleList()
        for i in range(1, orders+1):
            self.gcnList.append(SageGCN(input_dim//i, input_dim//(2*i)))
        self.cls_model = Classification(input_dim//(2*orders), output_dim)

    def forward(self, node_features):
        # input_dim = len(node_features[0][0]) 
        # orders = len(node_features) - 1 # n order neighbors
        # self.cls_model = Classification(1433, self.output_dim)#.to(util.device)

        # gcnList1 = []
        # for i in range(1, orders+1):
        #     gcnList1.append(SageGCN(input_dim//i, input_dim//(2*i)).to(util.device))
        # self.gcnList = nn.ModuleList(gcnList1)
        # cls_model = Classification(input_dim//(2*orders), self.output_dim)#.to(util.device)

        for order in range(self._orders):
            new_features = []
            for i in range(len(node_features) - 1):
                cur_features = node_features[i]
                neighbor_features = util.trans_view(node_features[i+1], (len(cur_features), -1, len(cur_features[0])))
                new_features.append(self.gcnList[order](cur_features, neighbor_features))
            node_features = new_features
        prediction = self.cls_model(node_features[0])
        
        return prediction

# for class test
if __name__ == '__main__':
    # features = torch.Tensor(torch.randn(5,128)).to('cuda')
    # model = Classification(128,5).to('cuda')
    # prediction = model(features)
    # print(prediction)

    currents = torch.Tensor([
        [0.1,0.1,0.1],
        [0.1,0.1,0.1]
    ])
    neighbors = torch.Tensor([
        [
            [0.2,0.2,0.2],
            [0.1,0.2,0.1],
            [0.3,0.2,0.0]
        ],
        [
            [0.2,0.2,0.2],
            [0.1,0.2,0.1],
            [0.3,0.2,0.0]
        ]
    ])
    model = SageGCN(3,2)
    result = model(currents,neighbors)
    print(result)