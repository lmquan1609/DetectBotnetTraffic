import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import datetime
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils import *

PARENT_PATH = '.'
PREFIX_PATH = f'{PARENT_PATH}/CTU-13-Dataset'

def loadData(train_files):
    dfs = []
    for file in train_files:
        path = os.path.join(PREFIX_PATH, file)
        dfs.append(pd.read_csv(path))
    all_data = pd.concat(dfs)  # Concat all to a single df
    return all_data

def preprocessing(data):
    data.columns = data.columns.str.lower()

    data['target'] = data['label'].apply(convertLabel)
    data['starttime'] = pd.to_datetime(data['starttime'])

    # We replace Null value to Con 
    data['state'] = data.state.fillna(value='CON')

    # fill nan port forward 
    data['sport'] = data.sport.fillna(method='pad')
    data['dport'] = data.dport.fillna(method='pad')

    data['stos'] = data.stos.fillna(value=0.0)
    data['dtos'] = data.dtos.fillna(value=0.0)

    data = data.drop(data[data.stos == 192.0].index)
    return data

if __name__ == '__main__':
    test_files = ['capture20110811.binetflow','capture20110810.binetflow',\
        'capture20110816-2.binetflow','capture20110816.binetflow',\
        'capture20110818-2.binetflow','capture20110819.binetflow']

    train_files = []
    for file in os.listdir(PREFIX_PATH):
        if file in test_files: continue
        train_files.append(file)
    
    all_data = loadData(train_files)
    all_data = preprocessing(all_data)
    
    train, test = train_test_split(all_data, test_size=0.3, random_state=42)

    train_0, train_1 = train['target'].value_counts()[0] / len(train.index), train['target'].value_counts()[1] / len(train.index)
    test_0, test_1 = test['target'].value_counts()[0] / len(test.index), test['target'].value_counts()[1] / len(test.index)

    print("In Train: there are {} % of class 0 and {} % of class 1".format(train_0, train_1))
    print("In Test: there are {} % of class 0 and {} % of class 1".format(test_0, test_1))


    train.to_csv(f'{PARENT_PATH}/train_alldata_EDA.csv')
    test.to_csv(f'{PARENT_PATH}/test_alldata_EDA.csv')