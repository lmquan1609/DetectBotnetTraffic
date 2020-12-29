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
TRAIN_FILE_NAME = f'{PARENT_PATH}/train_alldata_EDA.csv'
TEST_FILE_NAME = f'{PARENT_PATH}/test_alldata_EDA.csv'

saved_dict = {}
PREPROCESSED_PATH = 'final'

def loadData(train):
    train.drop(columns='Unnamed: 0', axis=1, inplace=True)

    starttime = train['starttime']
    srcaddr = train['srcaddr']
    dstaddr = train['dstaddr']
    state = train['state']

    # Dropping columns which are not useful for the classification
    # label is for binary classification
    # all the other columns are address related and not present in sample train data
    train.drop(['starttime', 'srcaddr', 'dstaddr', 'state'], axis=1, inplace=True)

    # To use during test data transformation
    saved_dict['to_drop'] = ['starttime', 'srcaddr', 'dstaddr', 'state']

    # mode values of every features, will use to fill Null values of test
    mode_dict = train.mode().iloc[0].to_dict()

    saved_dict['moded_featute'] = mode_dict

    # creating x and y set from the dataset
    x_train, y_train, y_train_label = train.drop(columns=['label', 'target']), train['target'], train['label']

    print(x_train.shape, y_train.shape, y_train_label.shape)

    return x_train, y_train, y_train_label, mode_dict

def convertFeature(x_train):
    # getting categorical and numerical columns in 2 diff lists
    num_col = [ 'dur', 'sport', 'dport', 'totpkts', 'totbytes', 'srcbytes']
    cat_col = list(set(x_train.columns) - set(num_col))

    # To use later, during test data cleaning
    saved_dict['cat_col'] = cat_col
    saved_dict['num_col'] = num_col

    x_train['sport'] = x_train['sport'].apply(convertHexInSport)
    x_train['dport'] = x_train['dport'].apply(convertHexInSport)

def genPrepFiles(x_train, y_train, y_train_label, mode_dict):
    # Standardizing the data
    scaler = MinMaxScaler()
    num_col = [ 'dur', 'sport', 'dport', 'totpkts', 'totbytes', 'srcbytes']
    scaler = scaler.fit(x_train[num_col])

    x_train[num_col] = scaler.transform(x_train[num_col])

    # Onehot Encoding and Hashing Encoding

    stos_ = OneHotEncoder()
    dtos_ = OneHotEncoder()
    dir_ = OneHotEncoder()
    proto_ = OneHotEncoder()

    ohe_stos = stos_.fit(x_train.stos.values.reshape(-1,1))
    ohe_dtos = dtos_.fit(x_train.dtos.values.reshape(-1,1))
    ohe_dir = dir_.fit(x_train.dir.values.reshape(-1,1))
    ohe_proto = proto_.fit(x_train.proto.values.reshape(-1,1))

    for col, encoding in zip(['stos', 'dtos', 'dir', 'proto'], [ohe_stos, ohe_dtos, ohe_dir, ohe_proto]):
        x = encoding.transform(x_train[col].values.reshape(-1,1))
        tmp_df = pd.DataFrame(x.toarray(), dtype='int64', columns=[col+'_'+str(i) for i in encoding.categories_[0]])
        x_train = pd.concat([x_train.drop(col, axis=1), tmp_df], axis=1)

    if not os.path.exists(PREPROCESSED_PATH):
        os.makedirs(PREPROCESSED_PATH)

    pickle.dump(scaler, open(f'{PREPROCESSED_PATH}/scaler.pkl', 'wb'))  # Standard scaler
    pickle.dump(saved_dict, open(f'{PREPROCESSED_PATH}/saved_dict.pkl', 'wb'))  # Dictionary with important parameters
    pickle.dump(mode_dict, open(f'{PREPROCESSED_PATH}/mode_dict.pkl', 'wb'))  #  Dictionary with most frequent values of columns

    # Onehot encoder for categorical columns
    pickle.dump(ohe_proto, open(f'{PREPROCESSED_PATH}/ohe_proto.pkl', 'wb'))
    pickle.dump(ohe_dir, open(f'{PREPROCESSED_PATH}/ohe_dir.pkl', 'wb'))
    pickle.dump(ohe_stos, open(f'{PREPROCESSED_PATH}/ohe_stos.pkl', 'wb'))
    pickle.dump(ohe_dtos, open(f'{PREPROCESSED_PATH}/ohe_dtos.pkl', 'wb'))
    
    pickle.dump((x_train, y_train, y_train_label), open(f'{PREPROCESSED_PATH}/final_train.pkl', 'wb'))

if __name__ == '__main__':
    train = pd.read_csv(TRAIN_FILE_NAME)

    x_train, y_train, y_train_label, mode_dict = loadData(train)

    saveData(x_train, y_train, y_train_label, f'{PREPROCESSED_PATH}/final_train.pkl')

    convertFeature(x_train)

    genPrepFiles(x_train, y_train, y_train_label, mode_dict)