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
            InfoDatas.append(InfoData)
            InfoHeader=getHeader(InfoData)
            if InfoHeader not in InfoHeaders:
                InfoHeaders.append(InfoHeader)
    return InfoHeaders, InfoDatas

#获取需要的stage信息的列明
def get_stageHead():
    file="../eventLogs/inforHead/StageInfo.csv"
    data = pd.read_csv(file)
    data1 = data["SatgeInfo"]
    data_list = data1.values.tolist()
    return data_list

#将stage信息写入csv
def write_to_csv(filename,header,datas):
    with open(filename, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        writer.writerow(header)
        for row in datas:
            writer.writerow(row)
      
#获取详细信息              
def get_stage_table(data,to_file):
    StageHeaders, StageDatas=get_detailInfo("Stage Info",data)
    num=0
    datas = []
    dataset=[]
    for data in StageDatas:
        num=num+1
        if(num%2==0):
            datas.append(data)
        else:
            continue
    for dd in datas:
        data = []
        for head  in get_stageHead():
            data.append(dd[head])
        dataset.append(data)
    head = get_stageHead()
    write_to_csv(to_file,head,dataset)

#获取需要提取stage信息的log文件集合
def getFiles(file_dir):
    i = 1
    for root, dirs, files in os.walk(file_dir):  
        i += 1
    return files   

                 
if __name__=="__main__":
    filename=r"../eventLogs/Second_Stage/PageRank/log/"
    files = getFiles(filename)   
    # print(files) #当前路径下所有非目录子文件 
    for file in files:
        filename="../eventLogs/Second_Stage/PageRank/log/"+file
        to_file = "../eventLogs/Second_Stage/PageRank/stage/"+file+".csv"
        datas = getDictionary(filename)
        HeaderArrays,data = getHeaderArrays(datas)
        get_stage_table(data,to_file)