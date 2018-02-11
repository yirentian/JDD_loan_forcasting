import xgboost as xgb
from sklearn.model_selection import train_test_split
import random
import pickle
import warnings
random.seed(888)
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys
import json
import os
import datetime
import random
import time
import bisect
from scipy.sparse import coo_matrix

warnings.filterwarnings("ignore")

train_file = './data/training2.pkl'
data_set = pickle.loads(open(train_file,'rb').read(),encoding='iso-8859-1')
data_set.fillna(0.,inplace=True)

label = data_set['label'].values # ndarray

feature_list = list(data_set.columns)
feature_list.remove('uid')
#feature_list.remove('label')

training = data_set[feature_list].values

test_data = pickle.load(open('./data/test.pkl','rb'),encoding='iso-8859-1')
test_data.fillna(0.,inplace=True)
sub_df = test_data['uid'].copy()

kf = KFold(n_splits=5, random_state=2017, shuffle=True)
rmse_list = []
sub_pred = []

for train_index, val_index in kf.split(training):
    X_train, y_train, X_val, y_val = training[train_index], label[train_index], training[val_index], label[val_index]
    lr = Lasso(alpha=0.002, normalize=False, max_iter=100000, warm_start=True, precompute=True)
    model = lr.fit(X_train, y_train)
    rmse = mean_squared_error(y_val, lr.predict(X_val)) ** 0.5
    #print('train mae: %g' % rmse)
    print('train mae: %g' % rmse)


