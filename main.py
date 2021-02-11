from numpy.core.defchararray import mod
import models
import data_center as context
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import util

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE by Xuechen Zhao')
parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--epoch', type=int, default=6)
parser.add_argument('--epochSize', type=int, default=50)
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

util.device = args.device
dc = context.data_center(args.dataSet)

sample_num = [15, 10] # number of neighbor samples per order
input_dim = len(dc.data.features[0])
output_dim = len(dc.data.label_map)
orders = len(sample_num) # number of orders

model = models.GraphSage(input_dim, output_dim, orders).to(util.device)
criterion = nn.CrossEntropyLoss().to(util.device)
optimizer = optim.SGD(model.parameters(), lr=0.1)

def get_samples(type = context.set_type.trains):
    sample_nodes_index, sample_nodes_feature = dc.sampling(args.batchSize, sample_num, from_set=type)

    #features: contains n order neighbors
    sample_nodes_feature = [torch.from_numpy(samples).to(util.device) for samples in sample_nodes_feature]

    #labels: not contains neighbors
    sample_nodes_labels = dc.data.labels[sample_nodes_index]
    sample_nodes_labels = torch.from_numpy(sample_nodes_labels).to(util.device)

    return sample_nodes_labels, sample_nodes_feature

def train():
    model.train()
    for e in range(1,args.epoch+1):
        for i in range(args.epochSize):
            sample_nodes_labels, sample_nodes_feature = get_samples()

            prediction = model(sample_nodes_feature)
            loss = criterion(prediction, sample_nodes_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            corrects_index = prediction.max(1)[1]
            accuracy = torch.eq(corrects_index, sample_nodes_labels).float().mean().item()
            lossy = loss.item()
            print('Epoch:{:03d}, Batch:{:03d}, Loss:{:.4f}, Accuracy:{:.4f}'.format(\
                e, i, lossy, accuracy))
        
        eval(context.set_type.vals)
        eval(context.set_type.tests)

def eval(type = context.set_type.vals):
    model.eval()
    with torch.no_grad():
        sample_nodes_labels,sample_nodes_feature = get_samples(type)

        prediction = model(sample_nodes_feature)
        corrects_index = prediction.max(1)[1]
        accuracy = torch.eq(corrects_index, sample_nodes_labels).float().mean().item()
        print('{:5}==>accuracy:{:.4f}'.format(type.name, accuracy))

if __name__ == '__main__':
    train()


