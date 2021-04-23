# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:27:58 2021

@author: l
"""

import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import  OneHotEncoder
import numpy as np
import csv
from get_DAG import getdata

def get_Config(file_name):
    file_data = pd.read_csv(file_name)
    config_data = file_data.iloc[:, 1:-1].values

    enc = OneHotEncoder()
    # serializer = config_data[:,-5]
    # serializer = serializer.reshape(-1,1)
    supervise = config_data[:,-5]
    supervise = supervise.reshape(-1,1)
    useLegacyMode = config_data[:,-3]
    useLegacyMode = useLegacyMode.reshape(-1,1)
    # enc.fit(serializer)
    enc.fit(supervise)
    # 经过OneHot编码处理spark.serializer参数
    # temp = enc.transform(serializer).toarray()
    temp1 = enc.transform(supervise).toarray()
    # 经过OneHot编码处理spark.driver.maxResultSize参数
    enc.fit(useLegacyMode)
    temp2 = enc.transform(useLegacyMode).toarray()
    config_data1 = np.delete(config_data, [-5,-3], axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    config_data1 = min_max_scaler.fit_transform(config_data1)
    # config_data1 = np.delete(config_data1, -3, axis=1)
    config_data1 = np.hstack((config_data1,temp1))
    config_data1 = np.hstack((config_data1,temp2))

    # config_data = torch.from_numpy(config_data).float()
    # 经过MinMaxScaler标准化
    config_data1 = torch.from_numpy(config_data1).float()
    return config_data1



if __name__=="__main__":
    file_name = "../eventLogs/Pagerank/Pagerank700/pagerankSmall210120_data.csv"
    stage_file = "../eventLogs/Pagerank/Pagerank700/stage/"    
    config_data = get_Config(file_name)
    print(config_data[0:10])
    # print(data)