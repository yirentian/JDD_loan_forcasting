# enconding:utf8
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from vecstack import stacking
from math import sqrt
import os
import csv
import math
import numpy as np
import sys
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
import time
import warnings
from util import *


warnings.filterwarnings("ignore")

train_file = './data/training2.pkl'
data_set = pickle.load(open(train_file,'rb'),encoding='iso-8859-1')
data_set.fillna(0.,inplace=True)

label = data_set['label'].values # ndarray

feature_list = list(data_set.columns)
feature_list.remove('uid')
feature_list.remove('label')

training = data_set[feature_list].values

test_data = pickle.load(open('./data/test.pkl','rb'),encoding='iso-8859-1')
test_data.fillna(0.,inplace=True)
sub_df = test_data['uid'].copy()

del test_data['uid']
test_data = test_data.values

kf = KFold(n_splits = 5,random_state=2017,shuffle=True)
rmse_list = []
sub_pred = []

for train_index, val_index in kf.split(training):
    X_train, y_train, X_val, y_val = training[train_index], label[train_index],training[val_index],label[val_index]

    # lgb model
    reg = RandomForestRegressor(n_estimators=1000,n_jobs=-1)

    model=reg.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    MSE = sum(abs(y_val - y_pred)) / len(y_val)
    se = sum(abs(y_val - y_pred)) / len(y_val)
    print("rmse:",se)
    print("mse",MSE)
    rmse_list.append(rmse)



#print("kflod rmse: {}\n mean rmse : {}".format(rmse_list, np.mean(np.array(rmse_list))))

dataframe = pd.DataFrame(y_pred)

dataframe.to_csv("result-t2.csv",index=False,sep=',')

pred = np.mean(np.array(sub_pred),axis=0)
sub_df.loc[:,'pred'] = pred
sub_df.to_csv('submission.csv',sep=',',header=None,index=False,encoding='utf8')













