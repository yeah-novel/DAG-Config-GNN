# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:48:56 2020

@author: l
"""

import json
import numpy as np
import csv
import pandas as pd
import os   

##将文件json化然后转换成字典
def getDictionary(filename):
    f=open(filename,"r+",encoding = 'utf-8-sig')
    file = f.readlines()
    json_str = json.dumps(file,indent=4)
    data = json.loads(json_str)
    return data

##获取一个Header
def getHeader(data):
    a = []
    for headers in sorted(data.keys()):
        a.append(headers)
    return a

##获取字典的key组成文件的列
def getHeaderArrays(datas):
    headerss = []
    data2 = []
    for data in datas:
        data = json.loads(data)
        data2.append(data)
        a = getHeader(data)
        if not a in headerss:
            headerss.append(a)
    HeaderArrays = np.array(headerss)
    data3 = np.array(data2)
    return HeaderArrays,data3

def get_detailInfo(InfoName,datas):
    InfoDatas=[]
    InfoHeaders=[]
    for data in datas:
        if(InfoName in getHeader(data)):
            InfoData = data[InfoName]
            InfoData["StageId"]=data["Stage ID"]
            if(InfoData["Finish Time"]!=0 and "Task Type" in getHeader(data) ):
                InfoData["TaskType"]=data["Task Type"]
                InfoDatas.append(InfoData)
                InfoHeader=getHeader(InfoData)
                if InfoHeader not in InfoHeaders:
                    InfoHeaders.append(InfoHeader)
#    InfoHeaderss = np.array(InfoHeaders)
#    InfoDatass = np.array(InfoDatas)
    return InfoHeaders, InfoDatas

def get_stageHead():
    file="eventLogs/inforHead/TaskInfo.csv"
    data = pd.read_csv(file)
    data1 = data["TaskInfo"]
    data_list = data1.values.tolist()
    return data_list

def write_to_csv(filename,header,datas):
    with open(filename, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        writer.writerow(header)
        for row in datas:
            writer.writerow(row)
                    
def get_task_table(data,to_file):
    TaskHeaders, TaskDatas=get_detailInfo("Task Info",data)
#    print(TaskDatas[0])
#    print(TaskDatas[1])
    dataset=[]
    for dd in TaskDatas:
        data = []
        for head  in get_stageHead():
            data.append(dd[head])
#        print(data)
        dataset.append(data)
    head = get_stageHead()
    write_to_csv(to_file,head,dataset)
    
def getFiles():
    file_dir = r"eventLogs/Pagerank/ye_lan/Pagerank_large/log"
    i = 1
    for root, dirs, files in os.walk(file_dir):  
#        print(i)
        i += 1
    return files   

                
if __name__=="__main__":
    # files = getFiles()   
#    print(files) #当前路径下所有非目录子文件 
    # for file in files:
    #     filename="eventLogs/Pagerank/ye_lan/Pagerank_large/log/"+file
    #     to_file = "eventLogs/Pagerank/ye_lan/Pagerank_large/task/"+file+".csv"
    #     datas = getDictionary(filename)
    #     HeaderArrays,data = getHeaderArrays(datas)
    #     get_task_table(data,to_file)
    filename="eventLogs/PageRank/eventLogs-app-20201002213732-0007/app-20201002213732-0007"
    to_file = "eventLogs/PageRank/eventLogs-app-20201002213732-0007/task-app-20201002213732-0007.csv"
    datas = getDictionary(filename)
    HeaderArrays,data = getHeaderArrays(datas)
#    print(HeaderArrays[8])
#    print(getHeader(data[150]))
    get_task_table(data,to_file)