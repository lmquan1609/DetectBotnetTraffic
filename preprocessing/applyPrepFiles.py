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
from genPrepFiles import *

PARENT_PATH = '.'
TRAIN_FILE_NAME = f'{PARENT_PATH}/train_alldata_EDA.csv'
TEST_FILE_NAME = f'{PARENT_PATH}/test_alldata_EDA.csv'

saved_dict = {}
PREPROCESSED_PATH = f'{PARENT_PATH}/final'

def clean_data(data, saved_dict):
    '''
    Cleans given raw data. Performs various cleaning, removes Null and wrong values.
    Check for columns datatype and fix them.
    '''
    data.drop(columns='Unnamed: 0', axis=1, inplace=True)
    data.drop(columns=saved_dict['to_drop'], axis=1, inplace=True)
    
    # Cleaning the data
    data = data.drop(data[data['stos'] == 192.0].index)
    data['stos'] = data.stos.fillna(value=0.0)
    data['dtos'] = data.dtos.fillna(value=0.0)
    data.dropna(inplace=True) #drop sport and dport missing
    
    data['sport'] = data['sport'].apply(convertHexInSport)
    data['dport'] = data['dport'].apply(convertHexInSport)
    
    return data

def loadPrepFiles():
    # Parameters
    saved_dict = pickle.load(open(f'{PREPROCESSED_PATH}/saved_dict.pkl', 'rb'))
    # Mode value of all the columns
    mode_dict = pickle.load(open(f'{PREPROCESSED_PATH}/mode_dict.pkl', 'rb'))
    # Stanardscaler object
    scaler = pickle.load(open(f'{PREPROCESSED_PATH}/scaler.pkl', 'rb'))

    # One hot encoder objects
    ohe_dtos = pickle.load(open(f'{PREPROCESSED_PATH}/ohe_dtos.pkl', 'rb'))
    ohe_stos = pickle.load(open(f'{PREPROCESSED_PATH}/ohe_stos.pkl', 'rb'))
    le_dir = pickle.load(open(f'{PREPROCESSED_PATH}/ohe_dir.pkl', 'rb'))
    le_proto = pickle.load(open(f'{PREPROCESSED_PATH}/ohe_proto.pkl', 'rb'))

    return saved_dict, mode_dict, scaler, ohe_dtos, ohe_stos, le_dir, le_proto

def standardize(data, saved_dict, scaler):
    '''
    Stanardize the given data. Performs mean centering and varience scaling.
    Using stanardscaler object trained on train data.
    '''
    data[saved_dict['num_col']] = scaler.transform(data[saved_dict['num_col']])

def ohencoding(data, encoder):
    '''
    Onehot encoding the categoricla columns.
    Add the ohe columns with the data and removes categorical columns.
    Using Onehotencoder objects trained on train data.
    '''
    # Adding encoding data to original data
    for col, encoding in zip(['stos', 'dtos', 'dir', 'proto'], [encoder[2], encoder[3], encoder[0], encoder[1]]):
        x = encoding.transform(data[col].values.reshape(-1,1))
        tmp_df = pd.DataFrame(x.toarray(), dtype='int64', columns=[col+'_'+str(i) for i in encoding.categories_[0]])
        data = data.reset_index(drop=True).join(tmp_df)
        data = data.drop(col, axis=1)
    return data

if __name__ == '__main__':
    test = pd.read_csv(TEST_FILE_NAME)

    # Resetting index of test data
    test.reset_index(drop=True, inplace=True)

    # Cleaning data using clean_data()
    saved_dict, mode_dict, scaler, ohe_dtos, ohe_stos, le_dir, le_proto = loadPrepFiles()
    test = clean_data(test, saved_dict)

    # Standardscaling using stanardize()
    standardize(test, saved_dict, scaler)

    #Encoding using One-hot Encoding
    test = ohencoding(test, [le_dir, le_proto, ohe_stos, ohe_dtos])

    x_test, y_test, y_test_label = test.drop(columns=['label', 'target'], axis=1), test['target'], test['label']

    saveData(x_test, y_test, y_test_label, f'{PREPROCESSED_PATH}/final_test.pkl')