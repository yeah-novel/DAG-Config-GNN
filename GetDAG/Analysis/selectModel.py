# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:54:25 2021

@author: l
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def trainfun():
    train_file = "trainResult.csv"
    data = pd.read_csv(train_file)
    model1_loss = data['model1_loss'].values
    model2_loss = data['model2_loss'].values
    model3_loss = data['model3_loss'].values
    model4_loss = data['model4_loss'].values
    model3_loss = model3_loss[:100]
    model4_loss = model4_loss[:300]
    x1 = np.linspace(0,1000,1000)
    x2 = np.linspace(0,1000,len(model3_loss))
    x3 = np.linspace(0,1000,len(model4_loss))
    custom_lines = [Line2D([0], [0], c='r', alpha=0.3),
                    Line2D([0],[0], c='b', alpha=0.3),
                    Line2D([0],[0], c='g', alpha=0.3),
                    Line2D([0],[0], c='y', alpha=0.8),
                    ]
    fig, ax = plt.subplots()
    lines = ax.plot()
    ax.legend(custom_lines, ['DAG&Config', 'only Config', 'SVR', 'RandomForest'])
    plt.plot(x1, model1_loss, c='r', alpha=0.3)
    plt.plot(x1, model2_loss, c='b', alpha=0.3)
    plt.plot(x2, model3_loss, c='g', alpha=0.5)
    plt.plot(x3, model4_loss, c='y', alpha=0.5)
    plt.ylim(0, 90)
    plt.ylabel("train_loss")
    plt.title("Model Train")
    plt.xlabel("epoch")
    plt.show()
    
    model1_r2 = data['model1_r-square'].values
    model2_r2 = data['model2_r-square'].values
    model3_r2 = data['model3_r-square'].values
    model4_r2 = data['model4_r-square'].values
    model3_r2 = model3_r2[:100]
    model4_r2 = model4_r2[:100]
    custom_lines = [Line2D([0], [0], c='r', alpha=0.3),
                    Line2D([0],[0], c='b', alpha=0.3),
                    Line2D([0],[0], c='g', alpha=0.5),
                    Line2D([0],[0], c='y', alpha=0.5),
                    ]
    fig, ax = plt.subplots()
    lines = ax.plot()
    ax.legend(custom_lines, ['DAG&Config', 'only Config', 'SVR', 'RandomForest'])
    plt.plot(x1, model1_r2, c='r', alpha=0.3)
    plt.plot(x1, model2_r2, c='b', alpha=0.3)
    plt.plot(x2, model3_r2, c='g', alpha=0.5)
    plt.plot(x2, model4_r2, c='y', alpha=0.5)
    plt.ylim(0, 1)
    plt.ylabel("train_r2")
    plt.title("Model Train")
    plt.xlabel("epoch")
    plt.show()    

def testfun():
    test_file = "testResult.csv"
    data = pd.read_csv(test_file)
    model1_loss = data['model1_loss'].values
    model2_loss = data['model2_loss'].values
    model3_loss = data['model3_loss'].values
    x = np.linspace(0,200,20)
    custom_lines = [Line2D([0], [0], c='r', alpha=0.5),
                    Line2D([0],[0], c='b', alpha=0.5),
                    Line2D([0],[0], c='g', alpha=0.5),
                    ]
    fig, ax = plt.subplots()
    lines = ax.plot()
    ax.legend(custom_lines, ['DAG&Config', 'only Config'])
    plt.plot(x, model1_loss, c='r', alpha=0.5)
    plt.plot(x, model2_loss, c='b', alpha=0.5)
    plt.plot(x, model3_loss, c='g', alpha=0.5)
    plt.ylim(0, 80)
    plt.ylabel("test_loss")
    plt.title("Model Test")
    plt.xlabel("epoch")
    plt.show()
    
    model1_r2 = data['model1_r-square'].values
    model2_r2 = data['model2_r-square'].values
    model3_r2 = data['model3_r-square'].values

    x = np.linspace(0,200,20)
    custom_lines = [Line2D([0], [0], c='r', alpha=0.5),
                    Line2D([0],[0], c='b', alpha=0.5),
                    Line2D([0],[0], c='g', alpha=0.5),
                    ]
    fig, ax = plt.subplots()
    lines = ax.plot()
    ax.legend(custom_lines, ['DAG&Config', 'only Config', 'Config GNN'])
    plt.plot(x, model1_r2, c='r', alpha=0.5)
    plt.plot(x, model2_r2, c='b', alpha=0.5)
    plt.plot(x, model3_r2, c='g', alpha=0.5)
    plt.ylim(0.1, 1)
    plt.ylabel("test_r2")
    plt.title("Model Test")
    plt.xlabel("epoch")
    plt.show()  
    
def comparedBN():
    file1 = "WithoutBN.csv"
    file2 = "WithBN.csv"
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    data1_loss = data1['loss'].values
    data2_loss = data2['loss'].values
    
    x = np.linspace(0,len(data1_loss),len(data1_loss))
    plt.plot(x, data1_loss, color="r", linestyle="-", marker="*", linewidth=0.3)
    plt.plot(x, data2_loss, color="b", linestyle="-", marker="o", linewidth=0.3)
    plt.grid()
    plt.ylim(1, 5)
    plt.show()
    
if __name__ == "__main__":
    comparedBN()
    # trainfun()
    # testfun()
    