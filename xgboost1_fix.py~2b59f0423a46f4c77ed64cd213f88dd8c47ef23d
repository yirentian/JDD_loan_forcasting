# coding=utf-8
import pandas as pd
import xgboost as xgb
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

for train_index, val_index in kf.split(training):
    X_train, y_train, X_val, y_val = training[train_index], label[train_index], training[val_index], label[val_index]

    params = {
        'booster': 'gbtree',  # gbtree used
        #'objective': 'binary:logistic',
        'objective': 'reg:linear',
        'early_stopping_rounds': 50,
        'scale_pos_weight': 0.63,  # 正样本权重
        #'eval_metric': 'auc',
        'gamma': 0,
        'max_depth': 5,
        # 'lambda': 550,
        'subsample': 0.6,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'eta': 0.02,
        'seed': 12,
        'nthread': 3,
        'silent': 1
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dval, 'val_x'), (dtrain, 'train_x')]
    model = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)

    # 对测试集进行预测（以上部分之所以划分成验证集，可以用来调参）
    y_pred = model.predict(dval, ntree_limit=model.best_ntree_limit)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    print("rmse:", rmse)
    rmse_list.append(rmse)

