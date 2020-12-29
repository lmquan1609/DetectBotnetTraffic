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

from scipy.sparse import csr_matrix

from keras.models import Model
from keras.layers import Dense, Input, Activation
from keras.optimizers import SGD, Adam

from preprocessing.utils import *
from utilsModel import *
from modelDecisionTree import *

PARENT_PATH = '.'

# declare the path of Netflow file
PREFIX_PATH = f'{PARENT_PATH}/CTU-13-Dataset'
FIGURE_PATH = f'./Figures'

TRAIN_BOT_NAMES = ['Murlo', 'Virut', 'Rbot', 'Neris', 'Rbot', 'Virut', 'Rbot']
TEST_BOT_NAMES = ['Neris', 'Neris', 'Sogou', 'DonBot', 'Rbot', 'NSIS']

def loadData(trainfiles, trainBotnames):
    data = {}
    for name in set(trainBotnames):
        data[name] = pd.DataFrame()
    for file, name in zip(train_files, trainBotnames):
        if data[name].empty: 
            print("Create new botnet")
            data[name] = pd.read_csv(PREFIX_PATH + '/' + file)
        else: 
            print("Append botnet")
            data[name] = data[name].append(pd.read_csv(PREFIX_PATH + '/' + file))
    return data

def preprocess(dataFrame):
    for key in data:
        data[key].columns = data[key].columns.str.lower()    
        data[key]['target'] = data[key]['label'].apply(convertlabel)
        data[key] = data[key][data[key]['target'] == 1]
        data[key].drop(['starttime', 'srcaddr', 'dstaddr', 'state'], axis=1, inplace=True)
        data[key].drop(['label'], axis=1, inplace=True)
        data[key].reset_index(inplace=True)
        data[key].drop(data[key][data[key]['stos'] == 192.0].index)
        data[key].reset_index(inplace=True)
        data[key]['stos'] = data[key].stos.fillna(value=0.0)
        data[key]['dtos'] = data[key].dtos.fillna(value=0.0)
        data[key].dropna(inplace=True)
        data[key]['sport'] = data[key]['sport'].apply(convertHexInSport)
        data[key]['dport'] = data[key]['dport'].apply(convertHexInSport)
        data[key].drop(columns='level_0',axis=1, inplace=True)
        data[key].drop(columns='index',axis=1, inplace=True)
    for index, key in enumerate(data):
        data[key]['label'] = key
        data[key]['target'].replace(1, index, inplace=True)        
    df = pd.DataFrame()
    for key in data:
        df = df.append(data[key], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    return df

def normalize_data(data):
    num_col = [ 'dur', 'sport', 'dport', 'totpkts', 'totbytes', 'srcbytes']
    cat_col = list(set(data.columns) - set(num_col))
    
    cat_col.remove('target')
    cat_col.remove('label')
    
    scaler = MinMaxScaler()
    scaler = scaler.fit(data[num_col])
    data[num_col] = scaler.transform(data[num_col])
    
    pickle.dump(scaler, open('scaletrainBotnet.pkl', 'wb'))
    
    stos_, dtos_, dir_, proto_ = [OneHotEncoder() for _ in range(4)]

    ohe_stos = stos_.fit(data.stos.values.reshape(-1,1))
    ohe_dtos = dtos_.fit(data.dtos.values.reshape(-1,1))
    ohe_dir = dir_.fit(data.dir.values.reshape(-1,1))
    ohe_proto = proto_.fit(data.proto.values.reshape(-1,1))
    
    for col, encoding in zip(['stos', 'dtos', 'dir', 'proto'], [ohe_stos, ohe_dtos, ohe_dir, ohe_proto]):
        x = encoding.transform(data[col].values.reshape(-1,1))
        tmp_df = pd.DataFrame(x.toarray(), dtype='int64', columns=[col+'_'+str(i) for i in encoding.categories_[0]])
        data = pd.concat([data.drop(col, axis=1), tmp_df], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    target = data['target']
    label = data['label']
    data.drop('target', axis=1, inplace=True)
    data.drop('label', axis=1, inplace=True)
    return data,target, label

def getmodel(inputShape):
    inputs = Input(shape=inputShape)
    x = Dense(128, activation='relu')(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    outputs = Dense(4, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

if __name__ == '__main__':
    test_files = ['capture20110811.binetflow','capture20110810.binetflow',\
        'capture20110816-2.binetflow','capture20110816.binetflow',\
        'capture20110818-2.binetflow','capture20110819.binetflow']

    train_files = []
    for file in os.listdir(PREFIX_PATH):
        if file not in test_files: train_files.append(file)

    data = loadData(train_files, TRAIN_BOT_NAMES)
    data = preprocess(data)

    X, y, label = normalize_data(data)

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3)

    model = getmodel(14)

    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    encY = OneHotEncoder()
    trainY = encY.fit_transform(trainY.values.reshape(-1,1)).toarray()

    history = model.fit(trainX,trainY,epochs=30,verbose=2,validation_split=0.2,batch_size=128)

    historyPD = pd.DataFrame(history.history)
    historyPD.plot()
    plt.title('Plot of accuracy and loss')
    plt.savefig(f'{FIGURE_PATH}/Fig12.png')

    model.save(f'{FIGURE_PATH}/MLPv1.h5')