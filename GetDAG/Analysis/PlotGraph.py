# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:42:06 2021

@author: l
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

def ploty_yre(y_,y_pre,title):
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
    
def plot_loss():
    train_file1 = "trainResult1.csv"
    data = pd.read_csv(train_file1)
    model1_loss = data['loss1'].values
    train_file2 = "trainResult2.csv"
    data = pd.read_csv(train_file2)
    model2_loss = data['loss1'].values
    custom_lines = [Line2D([0], [0], c='r', alpha=0.5),
                Line2D([0],[0], c='b', alpha=0.5),
                ]
    x1 = np.linspace(0,len(model1_loss),len(model1_loss))
    x2 = np.linspace(0,len(model2_loss),len(model2_loss))
    # model3_loss = 0*x1+286.93
    # model4_loss = 0*x1+65.32
    fig, ax = plt.subplots()
    lines = ax.plot()
    ax.legend(custom_lines, ['DAG&Config', 'only Config'])
    plt.plot(x1, model1_loss, c='r', alpha=0.8)
    plt.plot(x2, model2_loss, c='b', alpha=0.8)
    # plt.plot(x1, model3_loss, c='g', alpha=0.5)
    # plt.plot(x1, model4_loss, c='y', alpha=0.5)
    # plt.ylim(0, 90)
    plt.ylabel("Train Loss")
    plt.title("Model Train")
    plt.xlabel("Iteration Times")
    plt.show()
    
def plot_r2():
    train_file1 = "trainResult1.csv"
    data = pd.read_csv(train_file1)
    model1_r2 = data['r-square'].values
    train_file2 = "trainResult2.csv"
    data = pd.read_csv(train_file2)
    model2_r2 = data['r-square'].values

    custom_lines = [Line2D([0], [0], c='r', alpha=0.3),
                Line2D([0],[0], c='b', alpha=0.3),
                Line2D([0],[0], c='g', alpha=0.3),
                Line2D([0],[0], c='y', alpha=0.8),
                ]
    x1 = np.linspace(0,len(model1_r2),len(model1_r2))
    x2 = np.linspace(0,len(model2_r2),len(model2_r2))
    model3_r2 = 0*x1+0.607
    model4_r2 = 0*x1+0.897
    fig, ax = plt.subplots()
    lines = ax.plot()
    ax.legend(custom_lines, ['DAG&Config', 'only Config', 'SVR', 'RandomForest'])
    plt.plot(x1, model1_r2, c='r', alpha=0.5)
    plt.plot(x2, model2_r2, c='b', alpha=0.5)
    plt.plot(x1, model3_r2, c='g', alpha=0.8)
    plt.plot(x1, model4_r2, c='y', alpha=0.8)
    plt.ylim(0, 1)
    plt.ylabel("Train Accuracy")
    plt.title("Model Train")
    plt.xlabel("Iteration Times")
    plt.show()

if __name__ == "__main__":
    plot_r2()