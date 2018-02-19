# coding=utf-8
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from tpot import TPOTRegressor
import numpy as np
warnings.filterwarnings("ignore")


train_file = './data/training2.pkl'
data_set = pickle.loads(open(train_file,'rb').read(),encoding='iso-8859-1')
data_set.fillna(0.,inplace=True)

label = data_set['label'].values # ndarray

feature_list = list(data_set.columns)
feature_list.remove('uid')
feature_list.remove('label')

training = data_set[feature_list].values

test_data = pickle.load(open('./data/test.pkl','rb'),encoding='iso-8859-1')
test_data.fillna(0.,inplace=True)
sub_df = test_data['uid'].copy()

# 训练模型并预测出结果

test_data = test_data.values
dtest=xgb.DMatrix(test_data)

kf = KFold(n_splits=2, random_state=2017, shuffle=True)
rmse_list = []
sub_pred = []

for train_index, val_index in kf.split(training):
    X_train, y_train, X_val, y_val = training[train_index], label[train_index], training[val_index], label[val_index]

    tpot = TPOTRegressor(generations=100, population_size=100, offspring_size=None,mutation_rate=0.9, crossover_rate=0.1,
                 scoring=None, cv=5, n_jobs=1,
                 max_time_mins=None, max_eval_time_mins=5,
                 random_state=None, config_dict=None, warm_start=False,
                 verbosity=0, disable_update_check=False)
    tpot.fit(X_train, y_train)
    tpot.score(X_val, y_val )

    tpot.export('tpot_exported_pipeline.py')