# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:20:05 2021

@author: l
"""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from torch_geometric.data import Data, DataLoader
from sklearn import preprocessing

#获取已经提取出来的stage文件集合
def getFiles(file_dir):
    i = 1
    for root, dirs, files in os.walk(file_dir):  
        i += 1
    return files  
   
#获取DAG边的信息
def getEdge(data_file):
    data = pd.read_csv(data_file)
    PStIDs_data = data['Parent IDs'].values
    StageID = data['Stage ID'].values
    N_Node = []
    N_Neis = []    
    edge_index = []
    i = 0
    for pstIds in PStIDs_data:
        j = 0
        str1 = pstIds.strip().split('[')[1]
        str2 = str1.strip().split(']')[0]
        ids = str2.strip().split(', ')
        for ID in ids:
            if(ID!=''):
                j = j+1
                N_Node.append(int(ID))
                N_Neis.append(int(StageID[i]))
        i=i+1
    N_Node = np.array(N_Node,dtype=np.int64)
    N_Neis = np.array(N_Neis,dtype=np.int64)
    edge_index.append(N_Node)
    edge_index.append(N_Neis)
    edge_index = np.array(edge_index,dtype=np.int64)
    edge_index = torch.from_numpy(edge_index)
    return edge_index

def chageData(x):
    for i in range(len(x)):
        if type(x[i])==str:
            if x[i].find("distinct")!=-1:
                x[i]=0
            elif x[i].find("flatMap")!=-1:
                x[i]=1
            elif x[i].find("runJob")!=-1:
                x[i]=2
    return x

def data_embed(data, num_embeddings,embedding_dim):
    embedding = nn.Embedding(num_embeddings, embedding_dim,padding_idx=0)
    data = embedding(data)
    return data
    
#获取顶点特征向量
def get_feature(data_file):
    data = pd.read_csv(data_file)
    stageName = data.iloc[:,2].values
    stageName = chageData(stageName)
    data['Stage Name'] =stageName
    stageName =data['Stage Name'].values
    stageTask = data.iloc[: ,-1: ].values
    stageTask = np.array(stageTask,dtype=np.int64)
    features = stageTask
    feature_dim = features.shape[1]
    StageID = data['Stage ID'].values
    feat_list = []
    i = 0
    for pstIds in StageID:
        feats = features[i]
        feat_list.append(feats)
        i=i+1
    feat_list = np.array(feat_list,dtype=np.float32)
    feat_Matrix = torch.from_numpy(feat_list)
    return feat_Matrix, feature_dim   
 
#划分类
def get_label(data_file):
    data = pd.read_csv(data_file)
    y = data.iloc[:, -1:].values
    randomPoint=[14.5 ,15.5, 17.5,20]
    ytag=[]
    for i in y:
        if i<randomPoint[0]:
            ytag.append(0)
        elif i<randomPoint[1]:
            ytag.append(1)
        elif i<randomPoint[2]:
            ytag.append(2)
        elif i<randomPoint[3]:
            ytag.append(3)
        else:
            ytag.append(4)
    return ytag, y

def getdata(config_file,stage_file):
    label_list, duration = get_label(config_file)
    file_dir = r"../eventLogs/Sort/stage/"
    files = getFiles(file_dir) 
    i = 0
    data_list = []
    for file in files:
        file = stage_file+file
        x, feature_dim = get_feature(file)
        edge_index = getEdge(file)
        # y = label_list[i]  #分类需要
        y = duration[i]   #回归
        # Data(x=None, edge_index=None, edge_attr=None, y=None, pos=None, normal=None, face=None, **kwargs)
        data = Data(x=x,y=y,edge_index = edge_index)
        data_list.append(data)
        i = i+1
    return data_list,feature_dim

if __name__=="__main__":
    file_name = "../eventLogs/Sort/Sort20210313145735.csv"
    stage_file = "../eventLogs/Sort/stage/"
    dag_data, feature_dim = getdata(file_name, stage_file)
    print(dag_data)
    # print(feature_dim)
