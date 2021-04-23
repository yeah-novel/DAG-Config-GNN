# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:44:26 2021

@author: l
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, r2_score
import csv
from Evaluate import r_square,Adj_r_square
from CNN import Config_nn

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
def get_data(file):
    data = pd.read_csv(file)
    conf = data.iloc[:,1:-5]
    print(conf.shape)
    config_data = ((conf-conf.min())/(conf.max()-conf.min())).values
    duration = data.iloc[:,-5].values
    return config_data, duration

 
    
# 划分训练集和测试集
def split_data(config_data,dag_data):
    X1_train, X1_test, X2_train, X2_test = train_test_split(config_data, dag_data, test_size=0.3)
    return X1_train, X1_test, X2_train, X2_test
       
def train(config_data, y,  config_dim):
    config_data = torch.from_numpy(config_data).float()
    ## X1 为dag数据，X2 为config数据
    X_train, X_test, y_train, y_test = split_data(config_data, y)
    print("训练集规模=====",X_train.shape)
    print("测试集规模=====",X_test.shape)
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float() 
    ############训练模型##############################
    net = Config_nn(config_dim)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.03)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()
    train_losses = []
    test_losses = []
    test_r2s = []
    train_r2s = []
    net.train()
    for t in range(5000):
        train_out = net(X_train)    # 喂给 net 训练数据 x, 输出预测值
        train_loss = loss_func(train_out, y_train)
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.detach().numpy())
        # train_losses.append(mean_squared_error(y_train.detach().numpy(), out.detach().numpy()))
        train_r2 = r2_score(y_train.detach().numpy(), train_out.detach().numpy())    
        train_r2s.append(train_r2)
        if (t+1) % 10==0:
            net.eval()
            test_out = net(X_test)
            test_loss = loss_func(test_out, y_test)
            optimizer.zero_grad()
            test_loss.backward()
            optimizer.step()
            test_losses.append(test_loss.detach().numpy())
            test_r2 = r2_score(y_test.detach().numpy(), test_out.detach().numpy())
            test_r2s.append(test_r2)
    y_1 = y_train.detach().numpy()
    y_pre1 = train_out.detach().numpy()
    plotGraph(y_1, y_pre1,"Model Train")
    y_2 = y_test.detach().numpy()
    y_pre2 = test_out.detach().numpy()
    plotGraph(y_2, y_pre2,"Model Test")
    return train_losses, train_r2s, test_losses, test_r2s

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
    x = np.linspace(0,330,330)
    plt.plot(x,x,'k')
    plt.show()
    
def write_to_csv(filename,header,datas):
    with open(filename, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        writer.writerow(header)
        for row in datas:
            writer.writerow(row) 
            
if __name__=="__main__":
    file_name = "../data/alldata.csv"
    config_data,y = get_data(file_name)
    y = y.reshape(-1,1)
    config_dim = config_data.shape[1]
    # print(config_data[:10])
    train_losses, train_r2s, test_losses, test_r2s = train(config_data,y, config_dim)   
    print(min(train_losses),max(train_r2s))
    print(min(test_losses),max(test_r2s))
    filename = "../result/trainResult2.csv"
    filename1 = "../result/testResult2.csv"
    dataset1 = {
        "loss1":train_losses,
        "r-square": train_r2s
        }
    dataset2 = {
        "loss1": test_losses,
        "r-square": test_r2s,
        }
    df1 = pd.DataFrame(dataset1)
    header1 = df1.columns
    datas1 = df1.values
    df2 = pd.DataFrame(dataset2)
    header2 = df2.columns
    datas2 = df2.values
    write_to_csv(filename, header1, datas1)
    write_to_csv(filename1, header2, datas2)


    


