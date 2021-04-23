# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:31:56 2021

@author: l
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import  mean_squared_error, r2_score  # 批量导入指标算法
import csv

def r_square(y_true, y_pred):
    a = np.square(y_pred - y_true)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1-b/e
    return f

def get_data(file_Name):
    data = pd.read_csv(file_Name)
    conf = data.iloc[:,1:-5]
    config_data = ((conf-conf.min())/(conf.max()-conf.min())).values
    duration = data.iloc[:,-5].values
    return config_data, duration

# 划分训练集和测试集
def split_data(X1,X2):
    X1_train, X1_test, X2_train, X2_test = train_test_split(X1, X2, test_size=0.3)
    return X1_train, X1_test, X2_train, X2_test

def train_SVR(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_train.shape)
    print(X_test.shape)
    svr_lin = SVR(kernel='rbf', C=1e3)
    svr_lin.fit(X_train,y_train)
    pre_train = svr_lin.predict(X_train)
    pre_test = svr_lin.predict(X_test)
    train_loss = mean_squared_error(pre_train, y_train)
    test_loss = mean_squared_error(pre_test, y_test)
    train_r2 = r_square(y_train, pre_train)
    test_r2 = r_square(y_test, pre_test)
    print("train_r-square:",train_r2,"\ntrain_loss:",train_loss)   
    print("test_r-square:",test_r2,"\ntest_loss:",test_loss)  
    plotGraph(y_train, pre_train,"Model Train")
    plotGraph(y_test, pre_test,"Model Test")

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
    plt.plot(x,x,'k')
    plt.show()
    
def write_to_csv(filename,header,datas):
    with open(filename, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        writer.writerow(header)
        for row in datas:
            writer.writerow(row)    
            
if __name__ == "__main__":
    file_Name = "../data/alldata.csv"
    config_data, duration = get_data(file_Name)
    # y = duration.reshape(-1,1)
    X_train, X_test, y_train, y_test = split_data(config_data, duration)
    train_SVR(config_data, duration)
    # print(min(train_losses),max(train_r2s))
    # print(min(test_losses),max(test_r2s))
    # filename = "trainResult3.csv"
    # filename1 = "testResult3.csv"
    # dataset1 = {
    #     "loss1":train_losses,
    #     "r-square": train_r2s
    #     }
    # dataset2 = {
    #     "loss1": test_losses,
    #     "r-square": test_r2s,
    #     }
    # df1 = pd.DataFrame(dataset1)
    # header1 = df1.columns
    # datas1 = df1.values
    # df2 = pd.DataFrame(dataset2)
    # header2 = df2.columns
    # datas2 = df2.values
    # write_to_csv(filename, header1, datas1)
    # write_to_csv(filename1, header2, datas2)