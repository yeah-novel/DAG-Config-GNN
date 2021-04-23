# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:27:59 2021

@author: l
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config_nn(nn.Module):
    def __init__(self,config_dim):
        super(Config_nn, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.linear1 = nn.Linear(config_dim,10)
        # self.bn_1 = nn.BatchNorm1d(10, momentum=0.5)
        self.linear2 = nn.Linear(10,16)
        # self.bn_2 = nn.BatchNorm1d(16, momentum=0.5)
        self.linear3 = nn.Linear(16,32)
        # self.bn_3 = nn.BatchNorm1d(32, momentum=0.5)
        self.linear4 = nn.Linear(32,64)
        # self.bn_4 = nn.BatchNorm1d(64, momentum=0.5)
        self.linear5 = nn.Linear(64,128)
        self.linear6 = nn.Linear(128,256)
        # self.bn_5 = nn.BatchNorm1d(128, momentum=0.5)
        # self.dropout = nn.Dropout(0.3)
        self.out_layer = nn.Linear(256,1)
    def forward(self, x):
        # x = F.relu(self.bn_1(self.linear1(x)))
        # x = F.relu(self.bn_2(self.linear2(x)))
        # x = F.relu(self.bn_3(self.linear3(x)))
        # x = F.relu(self.bn_4(self.linear4(x)))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        # x = F.relu(self.dropout(self.bn_4(self.linear4(x))))
        out = self.out_layer(x)
        return out