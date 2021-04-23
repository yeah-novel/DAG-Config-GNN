# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:35:00 2021

@author: l
"""
from sklearn.metrics import  mean_squared_error
import numpy as np


def Adj_r_square(rows_number,cols_number,r2):
    if rows_number - cols_number - 1 == 0:
        r2adj = 0
    else:
        r2adj = 1 - (1-r2)*(rows_number-1)/(rows_number - cols_number - 1)
    return r2adj


def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的R Squared"""
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)

def r_square(y_true, y_pred):
    a = np.square(y_pred - y_true)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1-b/e
    return f