# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:24:02 2021

@author: l
"""
import torch
from torch_geometric.nn import GCNConv,GlobalAttention,max_pool
from torch_scatter import  scatter_max
import torch.nn.functional as F
import torch.nn as nn

class GNNNet(torch.nn.Module):
    
    def __init__(self,feature_dim):
        super(GNNNet, self).__init__()
        self.bn_input = nn.BatchNorm1d(1, momentum=0.9) 
        self.conv1 = GCNConv(feature_dim, 8)
        self.bn_1 = nn.BatchNorm1d(8, momentum=0.9)
        self.conv2 = GCNConv(8, 16)
        self.bn_2 = nn.BatchNorm1d(16, momentum=0.9) 
        self.conv3 = GCNConv(16, 32)
        self.linear1 = torch.nn.Linear(32,64)
        self.linear2 = torch.nn.Linear(64,2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.bn_input(x)
        x = self.conv1(x, edge_index)
        x = self.bn_1(x)
        x = max_pool(x)
        x = self.conv2(x, edge_index)
        x = self.bn_2(x)
        x = max_pool(x)
        x = self.conv3(x, edge_index)
        x = max_pool(x)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        out = x
        return out
    
class Net(nn.Module):
    def __init__(self,feature_dim,config_dim):
        super(Net, self).__init__()
        self.GNNNet = GNNNet(feature_dim)
        self.linear1 = torch.nn.Linear(config_dim+2,25)
        self.linear2 = nn.Linear(25,32)
        self.linear3 = nn.Linear(32,64)
        self.linear4 = nn.Linear(64,128)
        self.dropout = nn.Dropout(0.5)
        self.out_layer = nn.Linear(128,1)
    def forward(self, dag_data, config_data):
        #DAG网络
        for i, batch in enumerate(dag_data):
            x1 = self.GNNNet(batch)
        #连接两个网络
        x = torch.cat((config_data, x1), 1) 
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.dropout(self.linear4(x)))
        out = self.out_layer(x)
        return out