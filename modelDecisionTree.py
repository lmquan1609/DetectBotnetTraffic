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
from sklearn.tree import DecisionTreeClassifier

from scipy.sparse import csr_matrix

from preprocessing.utils import *
from utilsModel import *

PARENT_PATH = './final'
TRAIN_FILE_NAME = f'{PARENT_PATH}/final_train.pkl'
TEST_FILE_NAME = f'{PARENT_PATH}/final_test.pkl'

saved_dict = {}
PREPROCESSED_PATH = 'final'

FIGURE_PATH = f'./Figures'

def loadDataAndPrepFiles():
    
    x_train, y_train, y_train_label = pickle.load(open(TRAIN_FILE_NAME, 'rb'))
    x_test, y_test, y_test_label = pickle.load(open(TEST_FILE_NAME, 'rb'))

    # Dictionaries
    saved_dict = pickle.load(open(f'{PARENT_PATH}/saved_dict.pkl', 'rb'))
    mode_dict = pickle.load(open(f'{PARENT_PATH}/mode_dict.pkl', 'rb'))

    # Standard scaler
    scaler = pickle.load(open(f'{PARENT_PATH}/scaler.pkl', 'rb'))

    # Onehot/Label encoders
    ohe_dtos = pickle.load(open(f'{PARENT_PATH}/ohe_dtos.pkl', 'rb'))
    ohe_stos = pickle.load(open(f'{PARENT_PATH}/ohe_stos.pkl', 'rb'))
    ohe_dir = pickle.load(open(f'{PARENT_PATH}/ohe_dir.pkl', 'rb'))
    ohe_proto = pickle.load(open(f'{PARENT_PATH}/ohe_proto.pkl', 'rb'))

    # Making the train data sparse matrix
    x_train_csr = csr_matrix(x_train.values)

    # Creating sparse dataframe with x_train sparse matrix
    x_train = pd.DataFrame.sparse.from_spmatrix(x_train_csr, columns=x_train.columns)

    # Making test data sparse matrix
    x_test_csr = csr_matrix(x_test.values)

    # Creating x_test sparse dataframe
    x_test = pd.DataFrame.sparse.from_spmatrix(x_test_csr, columns=x_test.columns)

    return x_train_csr, y_train, x_test_csr, y_test

if __name__ == '__main__':
    result_dict = {}

    x_train_csr, y_train, x_test_csr, y_test = loadDataAndPrepFiles()
    # DT classifier
    clf = DecisionTreeClassifier(class_weight={0:0.05, 1:0.95})
    param = {'max_depth':[8, 10, 12, 14],
            'min_samples_split':[2, 4, 6]}

    dt_clf = hyperparam_tuning(clf, param, x_train_csr, y_train, cv=3)

    # Plotting heatmap of scores with params
    result_visualization(dt_clf, param, title='Max depth and min sample split tuning', figureName=f'{FIGURE_PATH}/Fig6.png', param1='max_depth', param2='min_samples_split')

    print(f'Best estimator of Decision Tree: {dt_clf.best_estimator_}')

    # Tuning "min_samples_leaf" on top of best found params
    clf = dt_clf.best_estimator_
    param = {'min_samples_leaf':[9, 11, 13]}

    dt_clf = hyperparam_tuning(clf, param, x_train_csr, y_train,cv=3)
    result_visualization(dt_clf, param, title='Min sample lead tuning', figureName=f'{FIGURE_PATH}/Fig7.png', param1='min_samples_leaf')

    # Apply best value
    dt_param = {'max_depth': 14, 'min_samples_split': 6, 'min_samples_leaf':13}
    dt_best_clf = DecisionTreeClassifier(**dt_param)

    dt_clf, dt_auc, dt_f1, dt_far = evaluate_result(dt_best_clf, x_train_csr, y_train, x_test_csr, y_test, 'DT', 'Confusion matrix of Decision tree with tuned parameter', f'{FIGURE_PATH}/Fig8.png')

    # Saving the Model to disk
    print('-----Save decision tree model to file dt_clf.pkl------')
    pickle.dump(dt_clf, open(f'{PARENT_PATH}/dt_clf.pkl', 'wb'))

    # Saving scores of DT
    result_dict['name'] = ["DT"]
    result_dict['auc'] = [dt_auc]
    result_dict['f1'] = [dt_f1]
    result_dict['far'] = [dt_far]

    pickle.dump(result_dict, open(f'{PARENT_PATH}/result_decision_tree.pkl', 'wb'))