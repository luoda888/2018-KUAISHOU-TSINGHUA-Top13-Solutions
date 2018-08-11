import numpy as np
import pandas as pd 
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
from pandas import DataFrame as DF
import gc
from sklearn.preprocessing import LabelEncoder
import time
import networkx as nx
from sklearn.cluster import MeanShift,KMeans
# 1. 分别训练A,B榜数据，得到A，B模型，则只需要利用A榜模型预测B榜All Feature,得到预测值Model-A，将该列值并入Feature-B,即B榜维数加一，利用增强后的数据再训练Model-B
# 2. 尝试重编码User-id，合并A，B数据，得到Merge-AB，在此基础上提取特征，训练模型(但未知Video-id,Author-id是否为乱序)

reg_log_a = pd.read_csv('data/a/user_register_log.txt',sep='\t',header=None).rename(columns={0:'user_id',1:'day',2:'register_type',3:'device_type'})
aci_log_a = pd.read_csv('data/a/user_activity_log.txt',sep='\t',header=None).rename(columns={0:'user_id',1:'day',2:'page',3:'video_id',4:'author_id',5:'action_type'})
app_log_a = pd.read_csv('data/a/app_launch_log.txt',sep='\t',header=None).rename(columns={0:'user_id',1:'day'})
video_log_a = pd.read_csv('data/a/video_create_log.txt',sep='\t',header=None).rename(columns={0:'user_id',1:'day'})

reg_log_b = pd.read_csv('data/b/user_register_log.txt',sep='\t',header=None).rename(columns={0:'user_id',1:'day',2:'register_type',3:'device_type'})
aci_log_b = pd.read_csv('data/b/user_activity_log.txt',sep='\t',header=None).rename(columns={0:'user_id',1:'day',2:'page',3:'video_id',4:'author_id',5:'action_type'})
app_log_b = pd.read_csv('data/b/app_launch_log.txt',sep='\t',header=None).rename(columns={0:'user_id',1:'day'})
video_log_b = pd.read_csv('data/b/video_create_log.txt',sep='\t',header=None).rename(columns={0:'user_id',1:'day'})

reg_log = pd.concat([reg_log_a,reg_log_b],axis=0).reset_index(drop=True)
aci_log = pd.concat([aci_log_a,aci_log_b],axis=0).reset_index(drop=True)
app_log = pd.concat([app_log_a,app_log_b],axis=0).reset_index(drop=True)
video_log = pd.concat([video_log_a,video_log_b],axis=0).reset_index(drop=True)

reg_log.to_csv('data/user_register_log.txt',sep=' ',header=False,index=False)
aci_log.to_csv('data/user_activity_log.txt',sep=' ',header=False,index=False)
app_log.to_csv('data/app_launch_log.txt',sep=' ',header=False,index=False)
video_log.to_csv('data/video_create_log.txt',sep=' ',header=False,index=False)

print('可持久化 Finished...')

# Get Week
reg_log = reg_log[reg_log['device_type']!=1]
reg_log['week'] = reg_log['day'] % 7

def get_transform(now,start_date,end_date):
    get_trans = now[(now['day']>=start_date) & (now['day']<=end_date)]
    return get_trans

def get_label(start_date,end_date):
    merge_name = ['user_id','day']
    all_log = pd.concat([aci_log[merge_name],app_log[merge_name],video_log[merge_name]],axis=0)
    train_label = get_transform(all_log,start_date,end_date)
    train_1 = DF(list(set(train_label['user_id']))).rename(columns={0:'user_id'})
    train_1['label'] = 1
    reg_temp = get_transform(reg_log,start_date-16,start_date-1)
    train_1 = train_1[train_1['user_id'].isin(reg_temp['user_id'])]
    train_0 = DF(list(set(reg_temp['user_id'])-set(train_1['user_id']))).rename(columns={0:'user_id'})
    train_0['label'] = 0
    del train_label
    gc.collect()
    return pd.concat([train_1,train_0],axis=0) 

def check_id(uid,now):
    return now[now['user_id'].isin(uid)]

def get_category_count(name,deal_now,train_data,start_date,end_date):
    count = DF(deal_now.groupby(['user_id',name]).size().reset_index().rename(columns={0:'times'}))
    count_size = aci_log.groupby([name]).size().shape[0]
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
        train_data = pd.merge(train_data,deal_now[deal_now[name]==i].groupby(['user_id']).apply(lambda x:get_binary(x['day'],start_date,end_date)).reset_index().rename(columns={0:'binary_'+str(i)+'_'+name+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
        train_data = pd.merge(train_data,deal_now[deal_now[name]==i].groupby(['user_id']).apply(lambda x:get_time_log_weight_sigma(x['day'],start_date,end_date)).reset_index().rename(columns={0:'get_log_sigma_'+str(i)+'_'+name+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
        
        sum_data += train_data[new_name].values

    for i in range(0,count_size):
        new_name = 'see_'+name+'_'+str(i)
        train_data[new_name+'_ratio'] = train_data[new_name].values/sum_data

    return train_data

def get_binary_seq(now,start_date,end_date): 
    day = list(range(start_date,end_date+1))
    ans1 = 0
    binary_day = []
    for i in day:
        if i in now.unique():
            binary_day.append(1)
        else:
            binary_day.append(0)
    return binary_day

def get_binary(now,start_date,end_date): # Boss Feature
    ans = 0
    binary_day = get_binary_seq(now,start_date,end_date)
    for i in range(len(binary_day)):
        ans += binary_day[i]*(2**i)
    return ans

def get_binary_mol7(now,start_date,end_date):
    ans = 0
    binary_day = get_binary_seq(now,start_date,end_date)
    for i in range(len(binary_day)):
        ans += binary_day[i]*(2**(i%7))
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

def get_time_weight_sigma(now,start_date,end_date):
    window_len = end_date+1-start_date
    ans = np.zeros(window_len)
    sigma_ans = 0
    for i in now:
        ans[(i-1)%window_len] += 1
    for i in range(window_len):
        sigma_ans += ans[i]*(i+1)
    return sigma_ans

def get_id_feature(control,name,now,train_data,start_date,end_date):
    if end_date<start_date:
        return train_data
    
    now = get_transform(now,start_date,end_date)
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].nunique()).reset_index().rename(columns={0:'count_all_'+control+name+str(end_date-start_date)}).fillna(0),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].var()).reset_index().rename(columns={0:'see_var_'+control+name+str(end_date-start_date)}).fillna(0),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].mean()).reset_index().rename(columns={0:'see_mean_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].median()).reset_index().rename(columns={0:'see_median_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].mad()).reset_index().rename(columns={0:'see_mad_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].skew()).reset_index().rename(columns={0:'see_skew_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
   
    return train_data

def get_time_feature(control,name,now,train_data,start_date,end_date):
    if end_date<start_date:
        return train_data
    
    now = get_transform(now,start_date,end_date)

    t1 = time.time()
    # 描述性统计特征 10
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].nunique()).reset_index().rename(columns={0:'count_all_'+control+name+str(end_date-start_date)}).fillna(0),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].var()).reset_index().rename(columns={0:'see_var_'+control+name+str(end_date-start_date)}).fillna(0),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].mean()).reset_index().rename(columns={0:'see_mean_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].median()).reset_index().rename(columns={0:'see_median_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].mad()).reset_index().rename(columns={0:'see_mad_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].skew()).reset_index().rename(columns={0:'see_skew_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].kurt()).reset_index().rename(columns={0:'see_kurt_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].min()).reset_index().rename(columns={0:'see_min_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].max()).reset_index().rename(columns={0:'see_max_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:(x[name].max()-x[name].min())).reset_index().rename(columns={0:'see_max-min_'+control+name+str(end_date-start_date)}),on=['user_id'],how='left')
    t2 = time.time()
    print('Describe Feature Finished... Used: ',t2-t1)
    
    t1 = time.time()
    # 一阶差分 二阶差分 11
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().var()).reset_index().rename(columns={0:'diff_seq_var_'+control+name}).fillna(0),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().mean()).reset_index().rename(columns={0:'diff_seq_mean_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().median()).reset_index().rename(columns={0:'diff_seq_median_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().mad()).reset_index().rename(columns={0:'diff_seq_mad_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().skew()).reset_index().rename(columns={0:'diff_seq_skew_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().kurt()).reset_index().rename(columns={0:'diff_seq_kurt_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().min()).reset_index().rename(columns={0:'diff_seq_min_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().max()).reset_index().rename(columns={0:'diff_seq_max_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().max()-x[name].sort_values().diff().min()).reset_index().rename(columns={0:'diff_seq_max_gap_min_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().diff().min()).reset_index().rename(columns={0:'diff2_min_gap'+control+name+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:x[name].diff().diff().max()).reset_index().rename(columns={0:'diff2_max_gap'+control+name+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    t2 = time.time()
    print('Diff Feature Finished... Used: ',t2-t1)

    t1 = time.time()
    # FFT ori 5
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.var(np.fft.fft(x[name])))).reset_index().rename(columns={0:'fft_seq_var_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.mean(np.fft.fft(x[name])))).reset_index().rename(columns={0:'fft_seq_mean_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.median(np.fft.fft(x[name])))).reset_index().rename(columns={0:'fft_seq_median_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.max(np.fft.fft(x[name])))).reset_index().rename(columns={0:'fft_seq_mad_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.min(np.fft.fft(x[name])))).reset_index().rename(columns={0:'fft_seq_skew_'+control+name}),on=['user_id'],how='left') 
    t2 = time.time()
    print('FFT LAST ABS Feature Finished... Used: ',t2-t1)

    t1 = time.time()
    # FFT 01
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.var(np.fft.fft(get_binary_seq(x[name],start_date,end_date))))).reset_index().rename(columns={0:'fft_01seq_var_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.mean(np.fft.fft(get_binary_seq(x[name],start_date,end_date))))).reset_index().rename(columns={0:'fft_01seq_mean_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.median(np.fft.fft(get_binary_seq(x[name],start_date,end_date))))).reset_index().rename(columns={0:'fft_01seq_median_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.max(np.fft.fft(get_binary_seq(x[name],start_date,end_date))))).reset_index().rename(columns={0:'fft_01seq_mad_'+control+name}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:abs(np.min(np.fft.fft(get_binary_seq(x[name],start_date,end_date))))).reset_index().rename(columns={0:'fft_01seq_skew_'+control+name}),on=['user_id'],how='left') 
    t2 = time.time()
    print('FFT FIRST 01 ABS Feature Finished... Used: ',t2-t1)

    t1 = time.time()
    # 时间衰减 4
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_binary(x[name],start_date,end_date)).reset_index().rename(columns={0:'binary_'+control+name+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_binary_mol7(x[name],start_date,end_date)).reset_index().rename(columns={0:'binary_mol7'+control+name+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_time_log_weight_sigma(x[name],start_date,end_date)).reset_index().rename(columns={0:'get_log_sigma_'+control+name+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    train_data = pd.merge(train_data,now.groupby(['user_id']).apply(lambda x:get_time_weight_sigma(x[name],start_date,end_date)).reset_index().rename(columns={0:'get_sigma_'+control+name+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    print('SIGMA Feature Finished... Uesd: ',t2-t1)
    t2 = time.time()

    return train_data
    
def HowManyPeopleWatch(df):
    num_people = len(df['user_id'].unique())
    return num_people

def MostHandle(df):
    most_handle = df.groupby('video_id').size().max()
    return most_handle

def FavAuthorCreate(df):
    most_author = df.groupby('author_id').size().sort_values(ascending=False).index[0]
    create_video_num = len(df[df['author_id']==most_author]['video_id'].unique())
    watch_other_video_num = len(df[df['author_id']==most_author]['video_id'].unique())
    watch_other_video = 1 if watch_other_video_num>1 else 0
    return create_video_num , watch_other_video

def GongXianDu(df):
    d11 = df.set_index('video_id')
    d11['gongxian_rate'] = df.groupby('video_id').size() 
    d11['gongxian_rate'] = d11['gongxian_rate'] / d11['video_watched_times']
    meand = d11['gongxian_rate'].mean()
    sumd = d11['gongxian_rate'].sum()
    stdd = d11['gongxian_rate'].std()
    skeww = d11['gongxian_rate'].skew()
    kurtt = d11['gongxian_rate'].kurt()
    return sumd,meand,stdd,skeww,kurtt

def get_lx_day(now):
    k1 = np.array(now)
    k2 = np.where(np.diff(k1)==1)[0]
    i = 0
    ans = []
    while i<len(k2)-1:
        l1 = 1
        while k2[i+1]-k2[i]==1:
            l1 += 1
            i += 1
            if i == len(k2)-1:
                break
        if l1 == 1:
            i += 1
            ans.append(2)
        else:
            ans.append(l1+1)
    if len(k2)==1:
        ans.append(2)
    return ans

def get_lx_day_feature(name,train_data,now):
    lx_now = now.groupby(['user_id']).apply(lambda x:get_lx_day(x['day'].sort_values(ascending=True).unique())).reset_index().rename(columns={0:'lx_day'})
    lx_collect = {
        name+'lx_count_len' : [],
        name+'lx_max' : [],
        name+'lx_min' : [],
        name+'lx_var' : []
    }

    if (lx_now.shape[0]==0) | (lx_now.shape[1]==0):
        lx_collect[name+'lx_count_len'].append(0)
        lx_collect[name+'lx_max'].append(0)
        lx_collect[name+'lx_min'].append(0)
        lx_collect[name+'lx_var'].append(-1)
    else:
        for i in lx_now['lx_day'].values:
            lx_collect[name+'lx_count_len'].append(len(i))
            lx_collect[name+'lx_max'].append(np.max(i))
            lx_collect[name+'lx_min'].append(np.min(i))
            lx_collect[name+'lx_var'].append(np.var(i))
    lx_collect = DF(lx_collect)
    lx_collect['user_id'] = lx_now['user_id']
    train_data = pd.merge(train_data,lx_collect,on=['user_id'],how='left')
    return train_data

def get_only_user_author_graph_feature(now,name,train_data):
    G = nx.DiGraph()
    need_to = ['user_id','author_id']
    to_make = now[need_to].drop_duplicates()
    for edge in to_make[need_to].values:
        G.add_edge(edge[0],edge[1])
    
    pr = nx.pagerank(G,alpha=0.8)
    G_degree1 = DF(dict(G.degree),index=['G_degree'+name]).T.reset_index().rename(columns={'index':'user_id'}).fillna(0)
    G_indegree1 = DF(dict(G.in_degree),index=['G_indegree'+name]).T.reset_index().rename(columns={'index':'user_id'}).fillna(0)
    G_pr1 = DF(pr,index=['G_pagerank'+name]).T.reset_index().rename(columns={'index':'user_id'}).sort_values(by=['G_pagerank'+name]).fillna(0)
   
    train_data = pd.merge(train_data,G_degree1,on=['user_id'],how='left')
    train_data = pd.merge(train_data,G_indegree1,on=['user_id'],how='left')
    train_data = pd.merge(train_data,G_pr1,on=['user_id'],how='left')
    return train_data

def get_train(uid,start_date,end_date):
    
    t_start = time.time()
    
    t1 = time.time()
    
    train_act = check_id(uid,get_transform(aci_log,start_date,end_date))
    train_video = check_id(uid,get_transform(video_log,start_date,end_date))
    train_app = check_id(uid,get_transform(app_log,start_date,end_date))
    
    # Get Week
    train_act['week'] = (train_act['day'].values) % 7
    train_video['week'] = (train_video['day'].values) % 7
    train_app['week'] = (train_app['day'].values) % 7
    
    train_reg = reg_log[reg_log['user_id'].isin(uid)].rename(columns={'day':'reg_day'})
    train_act = pd.merge(train_act,train_reg[['user_id','reg_day']],on=['user_id'],how='left')
    train_video = pd.merge(train_video,train_reg[['user_id','reg_day']],on=['user_id'],how='left')
    train_app = pd.merge(train_app,train_reg[['user_id','reg_day']],on=['user_id'],how='left')
    
    train_act['aci_distance_from_reg'] = train_act['day'] - train_act['reg_day']
    train_video['video_distance_from_reg'] = train_video['day'] - train_video['reg_day']
    train_app['app_distance_from_reg'] = train_app['day'] - train_app['reg_day']

    del train_act['reg_day']
    del train_video['reg_day']
    del train_app['reg_day']
    gc.collect()
    
    train_act = train_act.sort_values(by=['user_id','day'],ascending=True)
    train_app = train_app.sort_values(by=['user_id','day'],ascending=True)
    train_video = train_video.sort_values(by=['user_id','day'],ascending=True)
    t2 = time.time()
    
    print(start_date,' To ',end_date,' Have User: ',len(uid))
    print('Data Prepare Use...',t2-t1)
    
    # Build
    train_data = DF()
    train_data['user_id'] = uid # 1 feature
    
    # 获取每个人的点击总数/动作类型总数 # 36 feature 
    t1 = time.time()
    
    train_act['at_page'] = (train_act['action_type']+100)*(train_act['page']+1)
    train_data = pd.merge(train_data,train_act.groupby(['user_id']).size().reset_index().rename(columns={0:'action_all_times'}),on=['user_id'],how='left').fillna(0)
    # Time Window  4 + 30 feature
    train_data = get_lx_day_feature('aci_',train_data,train_act)
    train_data = get_time_feature('aci_','day',train_act,train_data,start_date,end_date) 
    train_data = get_time_feature('aci_','aci_distance_from_reg',train_act,train_data,start_date,end_date) # 30
    for i in ['page','action_type','at_page']: # 4*5*6 120feature
        train_data = get_id_feature('aci_',i,train_act,train_data,start_date,end_date)

    t2 = time.time()
    print('Use Time: ',t2-t1,' Aci Finish... ','Shape: ',train_data.shape)
    
    # 获取每个人在Category中不同点击的分布 17 feature
    t1 = time.time() 
    train_data = get_category_count('page',train_act,train_data,start_date,end_date)
    train_data = get_category_count('action_type',train_act,train_data,start_date,end_date)
    t2 = time.time()
    print('Use Time: ',t2-t1,' Category Finish... ','Shape: ',train_data.shape)
    
    # 获取Video_log中的特征  27 feature
    t1 = time.time()
    
    train_data = get_lx_day_feature('video_',train_data,train_video)
    train_data = get_time_feature('video_','day',train_video,train_data,start_date,end_date) 
    train_data = pd.merge(train_data,train_video.groupby(['user_id']).apply(lambda x:((x['day'].mode().values[0])%7)).reset_index().rename(columns={0:'mode_week'+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    
    t2 = time.time()
    print('Use Time: ',t2-t1,' Video Finish... ','Shape: ',train_data.shape)
    
    # 获取App_log中的特征  27 feature
    t1 = time.time()
    
    train_data = get_lx_day_feature('app_',train_data,train_app)
    train_data = get_time_feature('app_','day',train_app,train_data,start_date,end_date) 
    train_data = get_time_feature('app_','app_distance_from_reg',train_app,train_data,start_date,end_date)
    train_data = pd.merge(train_data,train_app.groupby(['user_id']).apply(lambda x:((x['day'].mode().values[0])%7)).reset_index().rename(columns={0:'mode_week'+'_'+str(end_date-start_date)}),on=['user_id'],how='left')
    
    t2 = time.time() 
    print('Use Time: ',t2-t1,' App Finish... ','Shape: ',train_data.shape)
    
    # 获取注册类型特征 # 3 feature
    t1 = time.time()
    train_data = pd.merge(train_data,reg_log[['user_id','register_type','device_type','week','day']],on=['user_id'],how='left').rename(columns={'week':'reg_week','day':'reg_day'}) # 2
    train_data['rt_dt'] = (train_data['register_type']+1)*(train_data['device_type'])
    train_data['week_rt'] = (train_data['register_type']+1)*(train_data['reg_week']+1)
    train_data['week_dt'] = (train_data['device_type']+1)*(train_data['reg_week']+1)
    train_data['distance_reg_end_window'] = end_date-train_data['reg_day']+1
    train_data['distance_reg_start_window'] = train_data['reg_day']-start_date+1
    train_data['is_window_reg'] = 1 if train_data['reg_day'].all()>=start_date else 0
    # Need To Add 用户活跃时间与注册时间的差值
    
    t2 = time.time()
    print('Use Time: ',t2-t1,' Reg Finish... ','Shape: ',train_data.shape)
    
    # 获取业务逻辑特征  3 feature
    t1 = time.time()
    user_feature = DF(train_data['user_id'].unique())
    user_feature.columns = ['user_id']
    user_feature = user_feature.set_index('user_id')
    user_feature['HowManyPeople_Watch'] = train_act.groupby('author_id').apply(HowManyPeopleWatch)
    user_feature['Most_Handle'] = train_act.groupby('user_id').apply(MostHandle)
    
    #计算视频被观看总次数
    video_size = train_act.groupby('video_id').size().reset_index()
    video_size.columns = ['video_id','video_watched_times']
    train_act = pd.merge(train_act,video_size,on=['video_id'],how='left')

    #分别计算每个用户的贡献度和、均、方
    temp = train_act.groupby('user_id').apply(GongXianDu)
    user_feature['GongXianSum'] = temp.apply(lambda x:x[0])
    user_feature['GongXianMean'] = temp.apply(lambda x:x[1])
    user_feature['GongXianStd'] = temp.apply(lambda x:x[2])
    user_feature['GongXianSkeww'] = temp.apply(lambda x:x[3])
    user_feature['GongXianKurtt'] = temp.apply(lambda x:x[4])
    
    fav_author = train_act.groupby('user_id').apply(FavAuthorCreate)
    user_feature['FavAuthorCreate'] = fav_author.apply(lambda x:x[0])
    user_feature['WatchOtherVideo'] = fav_author.apply(lambda x:x[1])
    train_data = pd.merge(train_data,user_feature.reset_index(),on=['user_id'],how='left')
    
    t2 = time.time()
    print('Use Time: ',t2-t1,' User-Author Finish... ','Shape:',train_data.shape)
    
    # 获取聚类特征
    
    t1 = time.time()
    train_data = get_only_user_author_graph_feature(train_act,'',train_data)
    kmean = KMeans(n_clusters=20,n_jobs=20)
    train_data['cluster_graph'] = kmean.fit_predict(train_data[['G_degree','G_indegree','G_pagerank']].fillna(0))
    
    train_data = get_only_user_author_graph_feature(train_act[train_act['page']==0],'page0',train_data)

    t2 = time.time()
    print('Use Time: ',t2-t1,' Cluster Finish... ','Shape:',train_data.shape)
    
    # Feature End
    t_end = time.time()
    print('Get Feature Use All Time: ',t_end-t_start)
    
    return train_data

# offline
# 0 10day step 
# 1 14day step
# 2 16day step

print('The Style is ',style,'...')
print('Dealing Offline...')
style = 2

if style == 0:
    t1 = time.time()
    train_label = []
    train_data = []
    for i in range(1,8):
        try_label = get_label(i+10,i+16)
        train_label.append(try_label)
        train_data.append(get_train(try_label['user_id'],i,i+9))

    t2 = time.time()
    print('Deal Train Feature: ',t2-t1)

    t1 = time.time()
    valid_label = get_label(24,30)
    valid_data = get_train(valid_label['user_id'],14,23)
    t2 = time.time()
    print('Deal Valid Feature: ',t2-t1)
elif style == 1:
    t1 = time.time()
    train_label = []
    train_data = []
    for i in range(1,4):
        try_label = get_label(i+14,i+20)
        train_label.append(try_label)
        train_data.append(get_train(try_label['user_id'],i,i+13))

    t2 = time.time()
    print('Deal Train Feature: ',t2-t1)

    t1 = time.time()
    valid_label = get_label(24,30)
    valid_data = get_train(valid_label['user_id'],10,23)
    t2 = time.time()
    print('Deal Valid Feature: ',t2-t1)
    
elif style == 2:
    t1 = time.time()
    train_label = get_label(17,23)
    train_data = get_train(train_label['user_id'],1,16)
    valid_label = get_label(24,30)
    valid_data = get_train(valid_label['user_id'],8,23)
    t2 = time.time()
    print('Deal Train Feature: ',t2-t1)

# online
print('Dealing Online...')
if style == 0:
    t1 = time.time()
    online_label = []
    online_data = []
    for i in range(8,15):
        try_label = get_label(i+10,i+16)
        online_label.append(try_label)
        online_data.append(get_train(try_label['user_id'],i,i+9))

    # model
    online_test = get_train(reg_log['user_id'].unique(),21,30)
    t2 = time.time()

    print('Deal Online: ',t2-t1)
elif style == 1:
    t1 = time.time()
    online_label = []
    online_data = []
    for i in range(4,11):
        try_label = get_label(i+14,i+20)
        online_label.append(try_label)
        online_data.append(get_train(try_label['user_id'],i,i+13))

    # model
    online_test = get_train(reg_log['user_id'].unique(),17,30)
    t2 = time.time()

    print('Deal Online: ',t2-t1)
elif style == 2:
    # online
    online_label = pd.concat([train_label,valid_label],axis=0).reset_index(drop=True)
    online_data =  pd.concat([train_data,valid_data],axis=0).reset_index(drop=True)

    # model
    online_test_b = get_train(reg_log_b['user_id'].unique(),15,30)

def merge_data(df):
    return pd.concat(df,axis=0).reset_index(drop=True)

if style!=2 :
    train_data = merge_data(train_data)
    train_label = merge_data(train_label)
    online_data = merge_data(online_data)
    online_label = merge_data(online_label)

    online_data = pd.concat([train_data,online_data],axis=0).reset_index(drop=True)
    online_label = pd.concat([train_label,online_label],axis=0).reset_index(drop=True)

path_name = 'pre_data/style_'+str(style)
train_data.to_csv(path_name+'/train_data.csv',index=False)
train_label.to_csv(path_name+'/train_label.csv',index=False)
valid_data.to_csv(path_name+'/valid_data.csv',index=False)
valid_label.to_csv(path_name+'/valid_label.csv',index=False)
online_data.to_csv(path_name+'/online_data.csv',index=False)
online_label.to_csv(path_name+'/online_label.csv',index=False)
online_test.to_csv(path_name+'/online_test.csv',index=False)


# Need to Add : 
#   a. User-Author-Video Interfacing
#     1. Node2Vec 
#     2. User-Video/Author Embedding # 2
#     3. User-Author-Video Embedding # 1
#     4. User-Video/Author Tf-idf/Word2Vec  # 2*2
#     5. User-Video/Author Cluster (By Tf-idf/Word2Vec) # 2*2
#     6. User-Author-Video Embedding/Tf-idf/Word2Vec + Cluster
#   b. Know More Author
#     1. Define "An Active User", For Example, You can choose "All of the Positive Sample" using their Mean Value
#     (Tips: Mean Value is the Times of Watching Video,Looking Author,See Page,Action Click Count)
#     2. Get Diff Value For Active User
#     3. Calc The UV metric , Fuv or Iuv. 
#     (Tips: Find top100 the most active User,Get their favourite Author/Video Union Set, Ex https://zhuanlan.zhihu.com/p/20943978)
#     4. Node Centrality/Influence (Wiki)
#     5. See Author Delay

