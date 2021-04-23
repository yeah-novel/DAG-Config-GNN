# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:46:55 2021

@author: l
"""

import numpy as np
import pandas as pd
import os
import csv

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
    # print(PStIDs_data)
    # print(StageID)
    N_Node = ""
    N_Neis = ""
    # edge_index = []
    i = 0
    for pstIds in PStIDs_data:
        j = 0
        str1 = pstIds.strip().split('[')[1]
        str2 = str1.strip().split(']')[0]
        ids = str2.strip().split(', ')
        for ID in ids:
            if(ID!=''):
                j = j+1
                N_Node = N_Node + ID + ''
                N_Neis = N_Neis + str(StageID[i]) + ''
        i=i+1
    # print(str(N_Node).split(" ")[:-1])    
    return N_Node, N_Neis

#获取顶点特征向量
def get_feature(data_file):
    data = pd.read_csv(data_file)
    stageTask = data.iloc[: ,-1: ].values
    stageTask = np.array(stageTask,dtype=np.int64)
    features = stageTask
    feature_dim = features.shape[1]
    StageID = data['Stage ID'].values
    feat_list = ""
    i = 0
    for pstIds in StageID:
        feat_list = feat_list + str(features[i][0]) + " "
        i=i+1
    return feat_list

#获取配置参数
def get_Config(file_name):
    file_data = pd.read_csv(file_name)
    head = file_data.columns[1:]
    duration = file_data.iloc[:,-1].values
    config_data = file_data.iloc[:, 1:-1].values
    
    return head, duration, config_data
    
def write_to_csv(filename,header,datas):
    with open(filename, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        writer.writerow(header)
        for row in datas:
            writer.writerow(row)

def mergeDAGConfig():
    stage_file = "../eventLogs/Second_Stage/PageRank/stage/"
    file_name = "../eventLogs/Second_Stage/PageRank/Pro_PageRank_data.csv"
    file_data = pd.read_csv(file_name)
    file_dir = r"../eventLogs/Second_Stage/PageRank/stage/"
    files = getFiles(file_dir) 
    i = 0
    N_Nodes = []
    N_Neises = []
    feats = []
    for file in files:
        file = stage_file+file
        N_Node, N_Neis = getEdge(file)
        feat_list = get_feature(file)
        N_Nodes.append(N_Node)
        N_Neises.append(N_Neis)
        feats.append(feat_list)
    file_data['N_Node'] = N_Nodes
    file_data['N_Neis'] = N_Neises
    file_data['Node_fetures'] = feats
    header = file_data.columns
    datas = file_data.values
    write_to_csv("second_stage/PageRank.csv", header, datas)  
    
if __name__=="__main__":
    # mergeDAGConfig()
    data1 = pd.read_csv("first_stage/WordCount.csv")
    data2 = pd.read_csv("first_stage/G_PageRank.csv")
    data3 = pd.read_csv("first_stage/Sort.csv")
    data4 = pd.read_csv("first_stage/PageRank.csv")
    data = data1.append(data2)
    data = data.append(data3)
    data = data.append(data4)
    header = data.columns
    datas = data.values
    write_to_csv("../data/alldata2.csv", header, datas)