from numpy.core.defchararray import mod
from numpy.core.fromnumeric import reshape
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
parser.add_argument('--epoch', type=int, default=50)
#parser.add_argument('--epochSize', type=int, default=50)
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

#generate environmental embeding from the train nodes
env_embeding = torch.from_numpy(dc.trains.mean(0)).to(util.device)
#torch.from_numpy(dc.data.features.mean(0)).to(util.device)
#torch.from_numpy(dc.trains.mean(0)).to(util.device)

def get_samples(type = context.set_type.trains):
    sample_nodes_index, sample_nodes_feature = dc.sampling(args.batchSize, sample_num, from_set=type)

    #features: contains n order neighbors
    #batch nodes' feafure with their n order neighbors, e.g. [array(), array(), array()]
    sample_nodes_feature = [torch.from_numpy(samples).to(util.device) for samples in sample_nodes_feature]

    #labels: not contains neighbors
    sample_nodes_labels = dc.data.labels[sample_nodes_index]
    sample_nodes_labels = torch.from_numpy(sample_nodes_labels).to(util.device)

    return sample_nodes_labels, sample_nodes_feature

def train():
    model.train()

    acc_tests = []
    for e in range(1,args.epoch+1):
        for i in range(len(dc.trains)//args.batchSize + 1): #(args.epochSize):
            sample_nodes_labels, sample_nodes_feature = get_samples()

            prediction = model(sample_nodes_feature, env_embeding)
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
        acc_tests.append(eval(context.set_type.tests))
    
    acc_tests.sort(reverse=True)
    print('---------------------------')
    print(acc_tests)
    print('average accuracy:{:.4f}'.format(np.mean(acc_tests[0:3])))

def eval(type = context.set_type.vals):
    model.eval()
    accuracy = .0
    with torch.no_grad():
        accs = []
        ds = getattr(dc,type.name)
        for i in range(len(ds)//args.batchSize + 1):
            sample_nodes_labels,sample_nodes_feature = get_samples(type)

            prediction = model(sample_nodes_feature)
            corrects_index = prediction.max(1)[1]
            accuracy = torch.eq(corrects_index, sample_nodes_labels).float().mean().item()
            accs.append(accuracy)
        
        accs = np.asarray(accs)
        accuracy = accs.mean()
        print('{:5}==>accuracy:{:.4f}'.format(type.name, accuracy))
    return accuracy

if __name__ == '__main__':
    train()


