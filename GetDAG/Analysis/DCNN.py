# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:24:02 2021

@author: l
"""
import torch
from torch_geometric.nn import GCNConv
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
        self.bn_3 = nn.BatchNorm1d(32, momentum=0.9) 
        self.linear1 = torch.nn.Linear(32,64)
        self.linear2 = torch.nn.Linear(64,2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.bn_input(x)
        x = self.conv1(x, edge_index)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn_3(x)
        x = F.relu(x)
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
        # self.bn_1 = nn.BatchNorm1d(25, momentum=0.5)
        self.linear2 = nn.Linear(25,32)
        # self.bn_2 = nn.BatchNorm1d(32, momentum=0.5)
        self.linear3 = nn.Linear(32,64)
        # self.bn_3 = nn.BatchNorm1d(64, momentum=0.5)
        self.linear4 = nn.Linear(64,128)
        # self.bn_4 = nn.BatchNorm1d(128, momentum=0.5)
        self.linear5 = nn.Linear(128,256)
        self.dropout = nn.Dropout(0.5)
        self.out_layer = nn.Linear(256,1)
    def forward(self, dag_data, config_data):
        #DAG网络
        for i, batch in enumerate(dag_data):
            x1 = self.GNNNet(batch)
        #连接两个网络
        x = torch.cat((config_data, x1), 1) 
        # x = F.relu(self.bn_1(self.linear1(x)))
        # x = F.relu(self.bn_2(self.linear2(x)))
        # x = F.relu(self.bn_3(self.linear3(x)))
        # x = F.relu(self.bn_4(self.linear4(x)))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        # x = F.relu(self.linear5(x))
        x = F.relu(self.dropout(self.linear5(x)))
        out = self.out_layer(x)
        return out