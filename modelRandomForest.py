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
from sklearn.ensemble import RandomForestClassifier

from scipy.sparse import csr_matrix

from preprocessing.utils import *
from utilsModel import *
from modelDecisionTree import *

PARENT_PATH = './final'
TRAIN_FILE_NAME = f'{PARENT_PATH}/final_train.pkl'
TEST_FILE_NAME = f'{PARENT_PATH}/final_test.pkl'

saved_dict = {}
PREPROCESSED_PATH = 'final'

FIGURE_PATH = f'./Figures'

if __name__ == '__main__':
    result_dict = {}

    x_train_csr, y_train, x_test_csr, y_test = loadDataAndPrepFiles()

    # Splitting train in train and cv data
    x_train_new_csr, x_cv_csr, y_train, y_cv = train_test_split(x_train_csr, y_train, test_size=0.2, random_state=42)

    # Tuning No of estimators
    param = {'n_estimators':[10, 15, 20]}
    rf_clf, param1, val1 = cross_validation(RandomForestClassifier, param, 'n_estimators', x_train_new_csr, y_train, x_cv_csr, y_cv, title='Number of estimator tuning', figureName=f'{FIGURE_PATH}/Fig9.png')

    # Tuning Criterion
    param = {'criterion':['gini', 'entropy'], 'min_samples_split':6, 'max_depth':14}
    dt_clf, _, _ = cross_validation(RandomForestClassifier, param, 'criterion', x_train_new_csr, y_train, x_cv_csr, y_cv, title='Criterion tuning', figureName=f'{FIGURE_PATH}/Fig10.png')

    # Best RandomForest model
    rf_bst_clf = RandomForestClassifier(criterion='entropy', max_depth=14, min_samples_split=6, n_estimators=20, n_jobs=-1)

    # Getting result on train and test data
    evaluate_result(rf_bst_clf, x_train_new_csr, y_train, x_test_csr, y_test, "RF", 'Confusion matrix of Decision tree with tuned parameter', f'{FIGURE_PATH}/Fig11.png')