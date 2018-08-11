# 该特征工程为不等长滑窗，滑窗的step为1天，全量数据

import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import gc
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
import time
from scipy import stats

register_log = pd.read_csv('/mnt/datasets/fusai/user_register_log.txt',sep='\t',header=None,dtype={0:np.uint32,1:np.uint8,2:np.uint16,3:np.uint16}).rename(columns={0:'user_id',1:'day',2:'register_type',3:'device_type'})
action_log = pd.read_csv('/mnt/datasets/fusai/user_activity_log.txt',sep='\t',header=None,dtype={0:np.uint32,1:np.uint8,2:np.uint8,3:np.uint32,4:np.uint32,5:np.uint8}).rename(columns={0:'user_id',1:'day',2:'page',3:'video_id',4:'author_id',5:'action_type'})
app_log = pd.read_csv('/mnt/datasets/fusai/app_launch_log.txt',sep='\t',header=None,dtype={0:np.uint32,1:np.uint8}).rename(columns={0:'user_id',1:'day'})
video_log = pd.read_csv('/mnt/datasets/fusai/video_create_log.txt',sep='\t',header=None,dtype={0:np.uint32,1:np.uint8}).rename(columns={0:'user_id',1:'day'})
register_log = register_log.sort_values(by=['user_id','day'],ascending=True)
action_log = action_log.sort_values(by=['user_id','day'],ascending=True)
app_log = app_log.sort_values(by=['user_id','day'],ascending=True)
video_log = video_log.sort_values(by=['user_id','day'],ascending=True)
register_log['week'] = register_log['day'] % 7
t1 = time.time()
app_log['diff_day'] = app_log.groupby(['user_id'])['day'].diff().fillna(-1)
app_log['diff_day'] = app_log['diff_day'].astype(np.int8)
t2 = time.time()
print('Diff APP Finished... ',t2-t1)
t1 = time.time()
video_log['diff_day'] = video_log.groupby(['user_id'])['day'].diff().fillna(-1)
video_log['diff_day'] = video_log['diff_day'].astype(np.int8)
t2 = time.time()
print('Diff Video Finished... ',t2-t1)
t1 = time.time()
action_log['diff_day'] = action_log.groupby(['user_id'])['day'].diff().fillna(-1)
action_log['diff_day'] = action_log['diff_day'].astype(np.int8)
t2 = time.time()
print('Diff Act Finished... ',t2-t1)

def reduce_mem_usage(props):
    # 计算当前内存
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of the dataframe is :", start_mem_usg, "MB")
    
    NAlist = []
    for col in props.columns:
        if (props[col].dtypes != object):
            isInt = False
            mmax = props[col].max()
            mmin = props[col].min()
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(-1, inplace=True) 
                props[col].replace(np.inf,-1,inplace=True)
            asint = props[col].fillna(-1).astype(np.int64)
            result = np.fabs(props[col] - asint)
            result = result.sum()
            if result < 0.01: 
                isInt = True
            if isInt:
                if mmin >= 0: 
                    if mmax <= 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mmax <= 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mmax <= 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else: 
                    if mmin > np.iinfo(np.int8).min and mmax < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mmin > np.iinfo(np.int16).min and mmax < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mmin > np.iinfo(np.int32).min and mmax < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mmin > np.iinfo(np.int64).min and mmax < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)  
            else: 
                props[col] = props[col].astype(np.float16)        
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props

def get_transform(now,start_date,end_date):
    get_trans = now[(now['day']>=start_date) & (now['day']<=end_date)]
    return get_trans

def get_label(start_date,end_date):
    merge_name = ['user_id','day']
    all_log = pd.concat([action_log[merge_name],app_log[merge_name],video_log[merge_name]],axis=0)
    train_label = get_transform(all_log,start_date,end_date)
    train_1 = DF(list(set(train_label['user_id']))).rename(columns={0:'user_id'})
    train_1['label'] = 1
    reg_temp = get_transform(register_log,1,start_date-1)
    train_1 = train_1[train_1['user_id'].isin(reg_temp['user_id'])]
    train_0 = DF(list(set(reg_temp['user_id'])-set(train_1['user_id']))).rename(columns={0:'user_id'})
    train_0['label'] = 0
    del train_label
    gc.collect()
    return pd.concat([train_1,train_0],axis=0) 

def check_id(uid,now):
    return now[now['user_id'].isin(uid)]
    
def get_mode(now):
    return stats.mode(now)[0][0]

def get_binary_seq(now,start_date,end_date): 
    day = list(range(1,end_date-start_date+2))
    ans1 = 0
    binary_day = []
    now_uni = now.unique()
    for i in day:
        if i in now_uni:
            binary_day.append(1)
        else:
            binary_day.append(0)
    return binary_day
    
def get_binary1(now,start_date,end_date): # Boss Feature
    ans = 0
    binary_day = get_binary_seq(now,start_date,end_date)
    for i in range(len(binary_day)):
        ans += binary_day[i]*(2**i)
    return ans

def get_binary2(now,start_date,end_date): # Boss Feature
    ans = 0
    binary_day = get_binary_seq(now,start_date,end_date)
    for i in range(len(binary_day)):
        ans += binary_day[i]*(1/(end_date-i))
    return ans

def get_time_log_weight_sigma(now,start_date,end_date):
    window_len = end_date+1-start_date
    ans = np.zeros(window_len)
    sigma_ans = 0
    for i in now:
        ans[(i-1)%window_len] += 1
    for i in range(window_len):
        if ans[i]!=0:
            sigma_ans += np.log(ans[i]/(window_len-i))
    return sigma_ans

def get_max_count(x,name):
    x_max = x[name].max()
    if x_max>0:
        return x[name].value_counts(x_max)
    else:
        return np.nan

def get_max_movie(x):
    x_max = x['day'].max()
    if x_max>0:
        x = x[x['day']==x_max]
        return x['video_id'].nunique()
    else:
        return np.nan

def get_type_feature(control,name,now,train_data,start_date,end_date,gap,gap_name):

    now = get_transform(now,start_date,end_date)
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:(end_date-x[name].max())).reset_index().rename(columns={0:'max1_'+control+name+gap_name}).fillna(end_date-start_date),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:(end_date-get_second_day(x[name],2))).reset_index().rename(columns={0:'max2_'+control+name+gap_name}).fillna(end_date-start_date),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_max_count(x,name)).reset_index().rename(columns={0:'max_count_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].nunique()).reset_index().rename(columns={0:'nunique_'+control+name+gap_name}).fillna(0),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_max_movie(x)).reset_index().rename(columns={0:'nunique_video_'+control+name+gap_name}).fillna(0),on=['user_id'],how='left')
    
    return train_data

def get_time_feature(control,name,now,train_data,start_date,end_date,gap,gap_name):
    
    now = get_transform(now,start_date,end_date)

    # 描述性统计特征 6
    t1 = time.time()
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].nunique()).reset_index().rename(columns={0:'nunique_all_'+control+name+gap_name}).fillna(0),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].count()/gap).reset_index().rename(columns={0:'count_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data['nunique / count' + control + name + gap_name] = train_data['nunique_all_'+control+name+gap_name] / train_data['count_'+control+name+gap_name]
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:(x[name].min()-start_date)).reset_index().rename(columns={0:'min-start_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:(end_date-x[name].min())).reset_index().rename(columns={0:'end-min_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_max_count(x,name)).reset_index().rename(columns={0:'max_count_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].quantile(q=0.75)/gap).reset_index().rename(columns={0:'q2_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].quantile(q=0.84)/gap).reset_index().rename(columns={0:'q3_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].quantile(q=0.96)/gap).reset_index().rename(columns={0:'q4_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_mode(x[name])).reset_index().rename(columns={0:'mode_'+control+name+gap_name}),on=['user_id'],how='left')
    
    t2 = time.time()
    print(name,' Describe Finished... ',t2-t1,' Shape: ',train_data.shape)
    
    t1 = time.time()
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_binary1(x[name],start_date,end_date)/gap).reset_index().rename(columns={0:'encoder1_01seq'+control+name+'_'+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_binary2(x[name],start_date,end_date)/gap).reset_index().rename(columns={0:'encoder2_01seq'+control+name+'_'+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_time_log_weight_sigma(x[name],start_date,end_date)/gap).reset_index().rename(columns={0:'LogSigma_'+control+name+'_'+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.std(np.fft.fft(x[name])))).reset_index().rename(columns={0:'fft_var_'+control+name+gap_name}).fillna(-1),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.mean(np.fft.fft(x[name])))).reset_index().rename(columns={0:'fft_mean_'+control+name+gap_name}),on=['user_id'],how='left')
    t2 = time.time()
    print(control,' Sigma Finished... ',t2-t1,' Shape: ',train_data.shape)
    
    t1 = time.time()
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:(end_date-x[name].max())).reset_index().rename(columns={0:'max1_'+control+name+gap_name}).fillna(end_date-start_date),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:(end_date-get_second_day(x[name],2))).reset_index().rename(columns={0:'max2_'+control+name+gap_name}).fillna(end_date-start_date),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:(end_date-get_second_day(x[name],3))).reset_index().rename(columns={0:'max3_'+control+name+gap_name}).fillna(end_date-start_date),on=['user_id'],how='left')
    
    t2 = time.time()
    
    print(control,' Max Finished... ',t2-t1,' Shape: ',train_data.shape)
    
    return train_data

def get_second_day(now,seq):
    now = list(now.unique())
    for i in range(seq-1):
        if len(now)>1:
            now.remove(max(now))
        else:
            return np.nan
    return max(now)

def get_id_feature(control,name,now,train_data,start_date,end_date,gap,gap_name):
    
    now = get_transform(now,start_date,end_date)

    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].count()/gap).reset_index().rename(columns={0:'count_'+control+name+gap_name}).fillna(0),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].nunique()).reset_index().rename(columns={0:'nunique_'+control+name+gap_name}).fillna(0),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].var()).reset_index().rename(columns={0:'var_'+control+name+gap_name}).fillna(0),on=['user_id'],how='left')
    
    return train_data

def get_diff_feature(control,name,now,train_data,start_date,end_date,gap,gap_name):
    
    now = get_transform(now,start_date,end_date)

    t1 = time.time()
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].count()).reset_index().rename(columns={0:'count_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].nunique()).reset_index().rename(columns={0:'nunique_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].std()).reset_index().rename(columns={0:'var_'+control+name+gap_name}).fillna(-1),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].mean()).reset_index().rename(columns={0:'mean_'+control+name+gap_name}).fillna(-1),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].max()).reset_index().rename(columns={0:'max_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_mode(x[name])).reset_index().rename(columns={0:'mode_'+control+name+gap_name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].min()).reset_index().rename(columns={0:'min_'+control+name+gap_name}),on=['user_id'],how='left')
    t2 = time.time()
    # print(control,' Get Diff Feature Finished... Used: ',t2-t1,' Shape: ',train_data.shape)
    return train_data

def get_category_count(name,deal_now,train_data,start_date,end_date):
    count = DF(deal_now.groupby(['user_id',name]).size().reset_index().rename(columns={0:'times'}))
    count_size = deal_now.groupby([name]).size().shape[0]
    sum_data = 0
    for i in range(0,count_size):
        new_name = 'see_'+name+'_'+str(i)
        temp = pd.merge(train_data,count[count[name]==i],on=['user_id']).rename(columns={'times':new_name})
        train_have = pd.merge(train_data,temp[['user_id',new_name]],on=['user_id'])
        train_have = train_have[['user_id',new_name]]
        not_have_name = list(set(train_data['user_id'].values)-set(train_have['user_id'].values))
        train_not_have = DF()
        train_not_have['user_id'] = train_data[train_data['user_id'].isin(not_have_name)]['user_id']
        train_not_have['see_'+name+'_'+str(i)] = 0
        temp = pd.concat([train_have,train_not_have],axis=0)
        train_data = pd.merge(train_data,temp,on=['user_id'],how='left')
        sum_data += train_data[new_name].values

    for i in range(0,count_size):
        new_name = 'see_'+name+'_'+str(i)
        train_data[new_name+'_ratio'] = train_data[new_name].values/sum_data

    return train_data

from multiprocessing import Pool
def parallelize_df_func(df, func, start, end, num_partitions=40, n_jobs=4):
    df_split = np.array_split(df, num_partitions)
    start_date = [start] * num_partitions
    end_date = [end] * num_partitions
    param_info = zip(df_split, start_date, end_date)
    pool = Pool(n_jobs)
    gc.collect()
    df = pd.concat(pool.map(func, param_info))
    pool.close()
    pool.join()
    return df
    
def get_train(param_info):
    uid = param_info[0]
    start_date= param_info[1]
    end_date= param_info[2]
    t_start = time.time()
    
    t1 = time.time()
    
    train_act = check_id(uid,get_transform(action_log,start_date,end_date))
    train_video = check_id(uid,get_transform(video_log,start_date,end_date))
    train_app = check_id(uid,get_transform(app_log,start_date,end_date))
    
    # Get Week
    train_act['week'] = (train_act['day'].values) % 7
    train_video['week'] = (train_video['day'].values) % 7
    train_app['week'] = (train_app['day'].values) % 7
    
    # Modify Day
    train_act['day'] = train_act['day'] - start_date + 1
    train_video['day'] = train_video['day'] - start_date + 1
    train_app['day'] = train_app['day'] - start_date + 1
    
    end_date = end_date-start_date+1
    true_start = start_date
    start_date = 1
    
    train_reg = register_log[register_log['user_id'].isin(uid)].rename(columns={'day':'reg_day'})
    train_act = pd.merge(train_act,train_reg[['user_id','reg_day']],on=['user_id'],how='left')
    train_video = pd.merge(train_video,train_reg[['user_id','reg_day']],on=['user_id'],how='left')
    train_app = pd.merge(train_app,train_reg[['user_id','reg_day']],on=['user_id'],how='left')
    
    del train_act['reg_day']
    del train_video['reg_day']
    del train_app['reg_day']
    gc.collect()
    
    t2 = time.time()
    
    print(start_date,' To ',end_date,' Have User: ',len(uid))
    print('Data Prepare Use...',t2-t1)
    
    # Build
    train_data = DF()
    train_data['user_id'] = uid # 1 feature
    
    train_data = pd.merge(train_data,train_act.groupby(['user_id']).size().reset_index().rename(columns={0:'action_all_times'}),on=['user_id'],how='left').fillna(0)
    for i in range(5):
        page_temp = train_act[train_act['page']==i]
        train_data = get_type_feature('act_page'+str(i),'day',page_temp,train_data,start_date,end_date,end_date-start_date,'_all')
    
    train_data = get_time_feature('act_','day',train_act,train_data,start_date,end_date,end_date-start_date,'_all')
    train_data = get_diff_feature('act_','diff_day',train_act,train_data,start_date,end_date,end_date-start_date,'_all')
    train_data = pd.merge(train_data,train_act.groupby(['user_id']).apply(lambda x:get_mode(x['week'])).reset_index().rename(columns={0:'act_mode_week'+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    
    for i in ['page','action_type','video_id','author_id']: # 4*3 12 feature
        train_data = get_id_feature('id_act_',i,train_act,train_data,start_date,end_date,end_date-start_date,'_all')
        
    train_data = get_category_count('page',train_act,train_data,start_date,end_date)
    train_data = get_category_count('action_type',train_act,train_data,start_date,end_date)
    train_data = get_category_count('page',train_act,train_data,end_date-3,end_date)
    train_data = get_category_count('action_type',train_act,train_data,end_date-3,end_date)

    train_data = reduce_mem_usage(train_data)
    train_data = get_time_feature('video_','day',train_video,train_data,start_date,end_date,end_date-start_date,'_all')  
    train_data = get_diff_feature('video_','diff_day',train_video,train_data,start_date,end_date,end_date-start_date,'_all') 
    train_data = reduce_mem_usage(train_data)
    train_data = get_time_feature('app_','day',train_app,train_data,start_date,end_date,end_date-start_date,'_all') 
    train_data = get_time_feature('app_','day',train_app,train_data,start_date,end_date,7,'_7') 
    train_data = get_diff_feature('app_','diff_day',train_app,train_data,start_date,end_date,end_date-start_date,'_all')
    train_data = pd.merge(train_data,train_app.groupby(['user_id']).apply(lambda x:get_mode(x['week'])).reset_index().rename(columns={0:'app_mode_week'+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    
    train_data = reduce_mem_usage(train_data)
    train_data = pd.merge(train_data,register_log[['user_id','register_type','device_type','week','day']],on=['user_id'],how='left').rename(columns={'week':'reg_week','day':'reg_day'}) # 2
    train_data['rt_dt'] = (train_data['register_type']+1)*(train_data['device_type']+1)
    train_data['week_rt'] = (train_data['register_type']+1)*(train_data['reg_week']+1)
    train_data['week_dt'] = (train_data['device_type']+1)*(train_data['reg_week']+1)
    t_end = time.time()
    print('Get Feature Use All Time: ',t_end-t_start)
    train_data = reduce_mem_usage(train_data)
    
    return train_data

def data_prepare(read_path=None):
    
    register_log = pd.read_csv(read_path+'user_register_log.txt',sep='\t',header=None,dtype={0:np.uint32,1:np.uint8,2:np.uint16,3:np.uint16}).rename(columns={0:'user_id',1:'day',2:'register_type',3:'device_type'})
    action_log = pd.read_csv(read_path+'user_activity_log.txt',sep='\t',header=None,dtype={0:np.uint32,1:np.uint8,2:np.uint8,3:np.uint32,4:np.uint32,5:np.uint8}).rename(columns={0:'user_id',1:'day',2:'page',3:'video_id',4:'author_id',5:'action_type'})
    app_log = pd.read_csv(read_path+'app_launch_log.txt',sep='\t',header=None,dtype={0:np.uint32,1:np.uint8}).rename(columns={0:'user_id',1:'day'})
    video_log = pd.read_csv(read_path+'video_create_log.txt',sep='\t',header=None,dtype={0:np.uint32,1:np.uint8}).rename(columns={0:'user_id',1:'day'})
    
    # Sort By User
    register_log = register_log.sort_values(by=['user_id','day'],ascending=True)
    action_log = action_log.sort_values(by=['user_id','day'],ascending=True)
    app_log = app_log.sort_values(by=['user_id','day'],ascending=True)
    video_log = video_log.sort_values(by=['user_id','day'],ascending=True)

    # Diff Day
    t1 = time.time()
    app_log['diff_day'] = app_log.groupby(['user_id'])['day'].diff().fillna(-1).astype(np.int8)
    video_log['diff_day'] = video_log.groupby(['user_id'])['day'].diff().fillna(-1).astype(np.int8)
    action_log['diff_day'] = action_log.groupby(['user_id'])['day'].diff().fillna(-1).astype(np.int8)
    t2 = time.time()
    print('Diff Day Finished... ',t2-t1)

    # Prepare REGISTER
    register_log['week'] = register_log['day'] % 7
    register_log['rt_dt'] = (register_log['register_type']+1)*(register_log['device_type']+1)
    register_log['week_rt'] = (register_log['register_type']+1)*(register_log['reg_week']+1)
    register_log['week_dt'] = (register_log['device_type']+1)*(register_log['reg_week']+1)
    register_log['use_reg_people'] = register_log.groupby(['register_type'])['user_id'].transform('count').values
    register_log['use_dev_people'] = register_log.groupby(['device_type'])['user_id'].transform('count').values
    register_log['week_rt_use_people'] = register_log.groupby(['week_rt'])['user_id'].transform('count').values
    register_log['week_dt_use_people'] = register_log.groupby(['week_dt'])['user_id'].transform('count').values
    register_log['rt_dt_use_people'] = register_log.groupby(['rt_dt'])['user_id'].transform('count').values

    return register_log,action_log,app_log,video_log

read_path = '/mnt/datasets/fusai/'
register_log,action_log,app_log,video_log = data_prepare(read_path)
train_set = []
for i in range(17,25):
    train_label = get_label(i,i+6)
    train_data_part1 = parallelize_df_func(train_label['user_id'], get_train, 1, i-1, 1, 1)
    train_data = pd.merge(train_data_part1,register_log[['user_id','use_reg_people','week','register_type','device_type','rt_dt',
                                                    'week_rt','week_dt','use_dev_people','week_rt_use_people','week_dt_use_people',
                                                    'rt_dt_use_people']],on=['user_id'],how='left')
    train_data = pd.merge(train_data,train_label,on=['user_id'],how='left')
    train_set.append(train_data)
    del train_data_part1
    gc.collect()

train_data = pd.concat(train_set[0:-1],axis=0).reset_index(drop=True)
valid_data = train_set[-1]
online_data = parallelize_df_func(register_log['user_id'].unique(), get_train, 1, 30, 1,1)
online_data = pd.merge(online_data,register_log[['user_id','use_reg_people','week','register_type','device_type','rt_dt',
                                            'week_rt','week_dt','use_dev_people','week_rt_use_people','week_dt_use_people',
                                            'rt_dt_use_people']],on=['user_id'],how='left')

write_path = '/home/kesci/'
train_data.to_csv(write_path+'train_data.csv',index=False)
valid_data.to_csv(write_path+'valid_data.csv',index=False)
online_data.to_csv(write_path+'online_data.csv',index=False)
print('Style 2 Feature Engineer Finished...')
