import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
from pandas import DataFrame as DF
import gc
import time
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Read
write_path = '/home/kesci/'
train_data = pd.read_csv(write_path+'train_data.csv')
valid_data = pd.read_csv(write_path+'valid_data.csv')
online_data = pd.read_csv(write_path+'online_data.csv')
online_train = pd.concat([train_data,valid_data],axis=0).reset_index(drop=True)

# LGB Model
feature_name = [i for i in train_data.columns if i not in ['user_id','label']]
print(len(feature_name))
xgb_train = xgb.DMatrix(train_data[feature_name],train_data['label'].values)
xgb_valid = xgb.DMatrix(valid_data[feature_name],valid_data['label'].values)
watch_list = [(xgb_train,'dtrain'),(xgb_valid,'dvalid')]

params = {
    'booster': 'gbtree',
    'objective': 'rank:pairwise', #'binary:logistic', 
    'eta': 0.05, 
    'seed' : 2018,
    'max_depth': 5,
    'subsample': 0.9, 
    'colsample_bytree': 0.8,
    'colsample_bylevel' : 0.8,
    'eval_metric': ['auc'], # Need TO Logloss
    'nthread' : 8,
    'gamma': 2,
}

xgb_model = xgb.train(params,xgb_train,2000,watch_list,early_stopping_rounds=40,verbose_eval=10)

pred = xgb_model.predict(xgb.DMatrix(valid_data[feature_name]))
from sklearn.metrics import roc_auc_score,f1_score
print('auc ',roc_auc_score(valid_data['label'],pred))
f1_ans = []
for i in pred:
    if i>=0.5:
        f1_ans.append(1)
    else:
        f1_ans.append(0)
print('f1 ',f1_score(valid_data['label'],f1_ans))

def create_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close()  

create_feature_map(feature_name)
import operator
xgb_importance = xgb_model.get_fscore(fmap='xgb.fmap')  
xgb_importance = sorted(xgb_importance.items(), key=operator.itemgetter(1))  
xgb_importance = DF(xgb_importance, columns=['name', 'fscore'])
print(xgb_importance)

online_xgb_set = xgb.DMatrix(online_train[feature_name],label=online_train['label'])
online_xgb_model = xgb.train(params,online_xgb_set,num_boost_round=xgb_model.best_iteration)
ans_xgb = online_xgb_model.predict(xgb.DMatrix(online_data[feature_name]))
submit_xgb = DF()
submit_xgb['id'] = online_data['user_id']
from sklearn.preprocessing import MinMaxScaler
st = MinMaxScaler()
submit_xgb['score'] = st.fit_transform(ans_xgb.reshape(-1,1)) # RANK
# submit_xgb['score'] = ans_xgb # Binary
print(submit_xgb.head(10))
print(submit_xgb['score'].describe())
submit.to_csv('Submit XGB.txt',index=False,header=False)


