# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:44:26 2021

@author: l
"""

import numpy as np
import torch
import pandas as pd
from torch_geometric.nn import GCNConv
from torch_scatter import  scatter_max
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold,train_test_split
from torch_geometric.data import Data, DataLoader
import csv
from Evaluate import r2_score,Adj_r_square
from DCNN import Net

    
# 将list类型转换成 tensor类型
def list2tensor(data_list):
    temp = np.array(data_list)
    return torch.from_numpy(temp)

def r_square(y_true, y_pred):
    a = np.square(y_pred - y_true)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1-b/e
    return f

# 获取数据
# def get_data(file):
#     data = pd.read_csv(file)
#     conf = data.iloc[:,1:-4]
#     config_data = ((conf-conf.min())/(conf.max()-conf.min())).values
#     dags = data.iloc[:,-3:].values
#     duration = data.iloc[:,-4].values
#     dag_data = []
#     for dag,y in zip(dags, duration):
#         edge_index = []
#         N_Node = [int(i) for i in dag[0].split(" ")[:-1] ]
#         N_Neis = [int(i) for i in dag[1].split(" ")[:-1] ]
#         feat_list = [[int(i)] for i in dag[2].split(" ")[:-1]]
#         N_Node = np.array(N_Node, dtype=np.int64)
#         N_Neis = np.array(N_Neis, dtype=np.int64)
#         edge_index.append(N_Node)
#         edge_index.append(N_Neis)
#         # print(N_Node)
#         edge_index = list2tensor(edge_index)
#         feat_list = list2tensor(feat_list).float()
#         data = Data(x=feat_list,y=[y],edge_index = edge_index)
#         dag_data.append(data)
#     return dag_data, config_data, duration

def get_data(file):
    data = pd.read_csv(file)
    conf = data.iloc[:,1:-5]
    config_data = ((conf-conf.min())/(conf.max()-conf.min())).values
    dags = data.iloc[:,-3:].values
    duration = data.iloc[:,-5].values
    # print(str(dags[0][0]).split(" ")[:])
    duration = data.iloc[:,-5].values
    dag_data = []
    for dag,y in zip(dags, duration):
        edge_index = []
        N_Node = [int(i) for i in str(dag[0]).split(" ")[0:-1] ]
        N_Neis = [int(i) for i in str(dag[1]).split(" ")[0:-1]]
        feat_list = [[int(i)] for i in dag[2].split(" ")[:-1]]
        N_Node = np.array(N_Node, dtype=np.int64)
        N_Neis = np.array(N_Neis, dtype=np.int64)
        edge_index.append(N_Node)
        edge_index.append(N_Neis)
        edge_index = list2tensor(edge_index)
        feat_list = list2tensor(feat_list).float()
        data = Data(x=feat_list,y=[y],edge_index = edge_index)
        dag_data.append(data)
    return dag_data

# 获取数据
def get_data1(file):
    data = pd.read_csv(file)
    dags = data.iloc[:,-3:].values
    duration = data.iloc[:,-5].values
    # print(str(dags[0][0]).split(" ")[:])
    dag_data = []
    for dag,y in zip(dags, duration):
        edge_index = []
        N_Node = [int(i) for i in str(dag[0]).split(" ")[0:]]
        N_Neis = [int(i) for i in str(dag[1]).split(" ")[0:]]
        feat_list = [[int(i)] for i in dag[2].split(" ")[:-1]]
        N_Node = np.array(N_Node, dtype=np.int64)
        N_Neis = np.array(N_Neis, dtype=np.int64)
        edge_index.append(N_Node)
        edge_index.append(N_Neis)
        edge_index = list2tensor(edge_index)
        feat_list = list2tensor(feat_list).float()
        data = Data(x=feat_list,y=[y],edge_index = edge_index)
        dag_data.append(data)
    return dag_data

def get_config(file_name):
    file_name1 = "../data/first_stage/WordCount.csv"
    # file_name2 = "../data/first_stage/G_PageRank.csv"
    file_name3 = "../data/first_stage/Sort.csv"
    file_name4 = "../data/first_stage/PageRank.csv"
    dag_data1= get_data1(file_name1)
    # dag_data2 = get_data(file_name2)
    dag_data3 = get_data1(file_name3)
    dag_data4= get_data(file_name4)
    dag_data = dag_data1+dag_data3+dag_data4
    data = pd.read_csv(file_name)
    conf = data.iloc[:,1:-5]
    config_data = ((conf-conf.min())/(conf.max()-conf.min())).values
    duration = data.iloc[:,-5].values
    return dag_data,config_data,duration

# 划分训练集和测试集
def split_data(config_data,dag_data):
    X1_train, X1_test, X2_train, X2_test = train_test_split(config_data, dag_data, test_size=0.3)
    return X1_train, X1_test, X2_train, X2_test
       
def train(config_data, dag_data, y, feature_dim, config_dim):
    config_data = torch.from_numpy(config_data).float()
    # print(config_data)
    ## X1 为dag数据，X2 为config数据
    X1_train, X1_test, X2_train, X2_test = split_data(dag_data, config_data) 
    y_train = [data.y for data in X1_train]
    y_test = [data.y for data in X1_test]
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    index_train = []
    X1_train = DataLoader(X1_train, batch_size=len(X1_train))
    X1_test = DataLoader(X1_test, batch_size=len(X1_test)) 
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    
    ############训练模型##############################
    net = Net(feature_dim, config_dim)       # 创建一个网络
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()           # 损失函数MSELoss
    train_losses = []                        # 训练时每次迭代的loss
    test_losses = []                         # 测试时的loss
    test_r2s = []                            # 测试时的r-square
    train_r2s = []                           # 训练时每次迭代的r-sqaure
    net.train()
    for t in range(3000):                    # 迭代次数3000
        train_out = net(X1_train, X2_train)  # 喂给 net 训练数据 x, 输出预测值
        train_loss = loss_func(train_out, y_train)  #获得预测值与真实值的误差
        optimizer.zero_grad()                # 清空上一步的残余更新参数值
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.detach().numpy())
        train_r2 = r2_score(y_train.detach().numpy(), train_out.detach().numpy())    
        train_r2s.append(train_r2)
        if (t+1) % 10==0:
            net.eval()                      # 模型测试
            test_out = net(X1_test, X2_test)
            test_loss = loss_func(test_out, y_test)
            optimizer.zero_grad()
            test_loss.backward()
            optimizer.step()
            test_losses.append(test_loss.detach().numpy())
            test_r2 = r2_score(y_test.detach().numpy(), test_out.detach().numpy())
            test_r2s.append(test_r2)
            
    torch.save(net, '\model12.pkl')
    print("train_r-square:",train_r2,"\ntrain_loss:",train_loss)   
    print("test_r-square:",test_r2,"\ntest_loss:",test_loss)    
    y_1 = y_train.detach().numpy()
    y_pre1 = train_out.detach().numpy()
    plotGraph(y_1, y_pre1,"Model Train")
    y_2 = y_test.detach().numpy()
    y_pre2 = test_out.detach().numpy()
    plotGraph(y_2, y_pre2,"Model Test")
    predicted = np.concentrate(y_pre1,y_pre2)
    true = y_1 + y_2
    return train_losses, train_r2s, test_losses, test_r2s, predicted, true

def plotGraph(y_,y_pre,title):
    bad_point = []
    good_point = []
    for i in range(len(y_)):
        if abs(y_pre[i]-y_[i])>10:
            bad_point.append(i)
        else:
            good_point.append(i)
    x1 = [y_[i] for i in bad_point]
    x2 = [y_[i] for i in good_point]
    y1 = [y_pre[i] for i in bad_point]
    y2 = [y_pre[i] for i in good_point]
    plt.scatter(x1, y1,alpha=0.3,c='r')   
    plt.scatter(x2, y2,alpha=0.3,c='b') 
    plt.title(title)
    plt.xlabel("true")   
    plt.ylabel("predict") 
    x = np.linspace(0,300,300)
    plt.plot(x,x,c='k')
    plt.show()
    
def write_to_csv(filename,header,datas):
    with open(filename, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        writer.writerow(header)
        for row in datas:
            writer.writerow(row) 
            
if __name__=="__main__":
    file_name = "../data/alldata.csv"
    data = pd.read_csv(file_name)
    # print(data)
    dag_data, config_data,y = get_config(file_name)
    print(len(dag_data))
    print(config_data.shape)
    # file_name = "../data/fixeddata.csv"
    # dag_data, config_data,y = get_data(file_name)
    # print(dag_data[650:1300])
    y = y.reshape(-1,1)
    feature_dim = 1
    config_dim = config_data.shape[1]
    # print(y)
    train_losses, train_r2s, test_losses, test_r2s, predicted, true= train(config_data, dag_data, y, feature_dim, config_dim)   
    filename1 = "../result/trainResult6.csv"
    filename2 = "../result/testResult6.csv"
    filename3 = "../result/result.csv"
    dataset1 = {
        "loss1":train_losses,
        "r-square": train_r2s,
        }
    dataset2 = {
        "loss1": test_losses,
        "r-square": test_r2s,
        }
    dataset3 = {
        "predicted":predicted,
        "true":true
        }
    df1 = pd.DataFrame(dataset1)
    header1 = df1.columns
    datas1 = df1.values
    df2 = pd.DataFrame(dataset2)
    header2 = df2.columns
    datas2 = df2.values
    df3 = pd.DataFrame(dataset3)
    header3 = df3.columns
    datas3 = df3.values
    write_to_csv(filename1, header1, datas1)
    write_to_csv(filename2, header2, datas2)
    write_to_csv(filename3, header3, datas3)


    


