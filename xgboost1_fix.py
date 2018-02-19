# coding=utf-8
import pandas as pd
import xgboost as xgb
import time
from sklearn import metrics
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
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

kf = KFold(n_splits=5, random_state=2017, shuffle=True)
rmse_list = []
sub_pred = []
start = time.time()
for train_index, val_index in kf.split(training):
    X_train, y_train, X_val, y_val = training[train_index], label[train_index], training[val_index], label[val_index]

    params = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'eta': 0.08,
        'num_round': 500, #300
        'max_depth': 3,
        'nthread': -1,
        'seed': 888,
        'silent': 1,
        'lambda':1500,
        'min_child_weight': 4
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dval, 'val_x'), (dtrain, 'train_x')]
    model = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)

    # 对测试集进行预测（以上部分之所以划分成验证集，可以用来调参）
    y_pred = model.predict(dval, ntree_limit=model.best_ntree_limit)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    MSE = sum(abs(y_val - y_pred)) / len(y_val)
    print("rmse:", rmse)
    print("MSE:",MSE)
    end = time.time()
    print (end - start)

    rmse_list.append(rmse)

