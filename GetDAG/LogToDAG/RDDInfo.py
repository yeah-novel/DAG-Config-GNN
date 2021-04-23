# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:48:56 2020

@author: l
"""

import json
import numpy as np
import csv
import pandas as pd
import StageTable as st

def get_rddHead():
    file="eventLogs/inforHead/RDDInfo.csv"
    data = pd.read_csv(file)
    data1 = data["RDDInfo"]
    data_list = data1.values.tolist()
    return data_list

def get_stageHead():
    file="eventLogs/inforHead/StageInfo.csv"
    data = pd.read_csv(file)
    data1 = data["SatgeInfo"]
    data_list = data1.values.tolist()
    return data_list

                    
def get_rdd_table(data,to_file):
    StageHeaders, StageDatas=st.get_detailInfo("Stage Info",data)
    #print(StageDatas)
    dataset=[]
    for stage in StageDatas:
        rddInfoes = stage['RDD Info']
        for rdd in rddInfoes:
            rdd['StageID']=stage['Stage ID']
            data = []
            for head  in get_rddHead():
                data.append(rdd[head])
            if(data not in dataset):
                dataset.append(data)
    head = get_rddHead()
    st.write_to_csv(to_file,head,dataset)  
    
                
if __name__=="__main__":
    # files = st.getFiles()
    # for file in files:
        filename="eventLogs/Pagerank/eventLogs-app-20201002213732-0007/app-20201002213732-0007"
        to_file = "eventLogs/Pagerank/eventLogs-app-20201002213732-0007/rdd-app-20201002213732-0007.csv"
        datas = st.getDictionary(filename)
        HeaderArrays,data = st.getHeaderArrays(datas)
        get_rdd_table(data,to_file)