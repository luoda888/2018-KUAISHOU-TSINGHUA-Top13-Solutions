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
dtrain = lgb.Dataset(train_data[feature_name], label=train_data['label'].values)
dval = lgb.Dataset(valid_data[feature_name], label=valid_data['label'].values)

params = {'learning_rate': 0.05,
          'metric': ['auc','binary_logloss'],
          'objective': 'binary',
          'nthread': 8,
          'num_leaves': 8,
          'colsample_bytree': 0.7,
          'bagging_fraction' : 0.8,
          'bagging_freq' : 10,
          'seed' : 2018,
        }
         
lgb_model = lgb.train(params, dtrain, 2500, dval, verbose_eval=50,early_stopping_rounds=100,)
pred = lgb_model.predict(train_data[feature_name])
from sklearn.metrics import roc_auc_score,f1_score
print('TRAIN SET auc ',roc_auc_score(train_data['label'],pred))
f1_ans = []
for i in pred:
    if i>=0.5:
        f1_ans.append(1)
    else:
        f1_ans.append(0)
print('TRAIN SET F1 ',f1_score(train_data['label'],f1_ans))

fi = DF()
fi['name'] = feature_name
fi['score'] = lgb_model.feature_importance()
print(fi.sort_values(by=['score'],ascending=False))
lgb.plot_importance(lgb_model,max_num_features=40,figsize=(10,8))
plt.show()

online_train = pd.concat([train_data,valid_data],axis=0).reset_index(drop=True)
online_lgb_set = lgb.Dataset(online_train[feature_name],label=online_train['label'])
online_lgb_model = lgb.train(params,online_lgb_set,num_boost_round=lgb_model.best_iteration-50)
ans = online_lgb_model.predict(online_data[feature_name])
submit = DF()
submit['id'] = online_data['user_id']
submit['score'] = ans
print(submit.head(10))
print(submit['score'].describe())
submit.to_csv('Submit.txt',index=False,header=False)


