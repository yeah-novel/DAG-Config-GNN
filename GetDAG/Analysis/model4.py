
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:42:47 2021

@author: l
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import  mean_squared_error  # 批量导入指标算法
import csv
from sklearn.model_selection import cross_val_score

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

def r_square(y_true, y_pred):
    a = np.square(y_pred - y_true)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1-b/e
    return f

def train_RF(X_train, X_test, y_train, y_test):
    regressor = RandomForestRegressor(n_estimators=100,max_features="sqrt",random_state=0)
#     cross_val_score(regressor, X_train, y_train, cv=10
# 				,scoring = "neg_mean_squared_error")
    # regr = RandomForestRegressor(n_estimators=2, max_depth=5)
    regressor.fit(X_train, y_train)
    pre_train = regressor.predict(X_train)
    train_r2 = regressor.score(X_train, y_train)
    train_mse = mean_squared_error(y_train, pre_train)
    pre_test = regressor.predict(X_test)
    test_r2 = regressor.score(X_test, y_test)
    test_mse = mean_squared_error(y_test, pre_test)
    print("train_r-square:",train_r2,"\ntrain_loss:",train_mse)   
    print("test_r-square:",test_r2,"\ntest_loss:",test_mse)  
    plotGraph(y_train, pre_train,"Model Train")
    plotGraph(y_test, pre_test,"Model Test")
    #     scores = cross_val_score(regr, X_train, y_train, cv=100
    #                ,scoring = "neg_mean_squared_error")
    # # sorted(sklearn.metrics.SCORERS.keys())
    #     print(scores)
   
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
    train_RF(X_train, X_test, y_train, y_test )
