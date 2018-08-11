import hashlib, math, os, subprocess
from multiprocessing import Process
import xlearn
import numpy as np
import pandas as pd
from padnas import DataFrame as DF

def hashstr(str, nr_bins=1e+6):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16) % (int(nr_bins) - 1) + 1

class FfmEncoder():
    def __init__(self, field_names, label_name, nthread=1):
        self.field_names = field_names
        self.nthread = nthread
        self.label = label_name

    def gen_feats(self, row):
        feats = []
        for field in self.field_names:
            value = row[field]
            key = field + '-' + str(value)
            feats.append(key)
        return feats

    def gen_hashed_fm_feats(self, feats):
        feats = ['{0}:{1}:1'.format(field, hashstr(feat, 1e+6)) for (field, feat) in feats]
        return feats

    def convert(self, df, path, i):
        lines_per_thread = math.ceil(float(df.shape[0]) / self.nthread)
        sub_df = df.iloc[i * lines_per_thread: (i + 1) * lines_per_thread]
        tmp_path = path + '_tmp_{0}'.format(i)
        with open(tmp_path, 'w') as f:
            for index,row in sub_df.iterrows():
                feats = []
                for i, feat in enumerate(self.gen_feats(row)):
                    feats.append((i, feat))
                feats = self.gen_hashed_fm_feats(feats)
                f.write(str(int(row[self.label])) + ' ' + ' '.join(feats) + '\n')

    def parallel_convert(self, df, path):
        processes = []
        for i in range(self.nthread):
            p = Process(target=self.convert, args=(df, path, i))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def delete(self, path):
        for i in range(self.nthread):
            os.remove(path + '_tmp_{0}'.format(i))

    def cat(self, path):
        if os.path.exists(path):
            os.remove(path)
        for i in range(self.nthread):
            cmd = 'cat {svm}_tmp_{idx} >> {svm}'.format(svm=path, idx=i)
            p = subprocess.Popen(cmd, shell=True)
            p.communicate()

    def transform(self, df, path):
        print('converting data......')
        self.parallel_convert(df, path)
        self.cat(path)
        self.delete(path)
        
write_path = '/home/kesci'
ffm_train = train_data.copy()
ffm_valid = valid_data.copy()
ffm_online_train = online_train.copy()
ffm_online_test = online_data.copy()
ffm_online_test['label'] = 0
# filed_names = list(fi.sort_values(by=['score'],ascending=False).head(50)['name'].values)
filed_names = [i for i in ffm_train.columns if i not in ['user_id','label']]
print(filed_names)
fe = FfmEncoder(filed_names,label_name='label',nthread=8)
fe.transform(ffm_train, write_path+'train.ffm')
print('Train FFM Finished...')
fe.transform(ffm_valid, write_path+'valid.ffm')
print('Valid FFM Finished...')
fe.transform(ffm_online_train,write_path+'train_online.ffm')
print('Train Online FFM Finished...')
fe.transform(ffm_online_test, write_path+'test_online.ffm')
print('Test Online FFM Finished')

# Training task
ffm_model = xl.create_ffm() # Use field-aware factorization machine
ffm_model.setTrain("/home/kesci/train.ffm")  # Training data
ffm_model.setValidate("/home/kesci/valid.ffm")  # Validation data

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy
param = {'task':'binary', 'lr':0.1,
         'lambda':0.01, 'metric':'auc', 'epoch' : 20,'opt':'ftrl'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, write_path+'model.out')
print('Offline Train Finished...')

# Prediction task
param = {'task':'binary', 'lr':0.05,
         'lambda':0.003, 'metric':'auc'}
ffm_online_model = xl.create_ffm()
ffm_online_model.setTrain(write_path+'train_online.ffm')
ffm_online_model.fit(param,write_path+'online_model.out')
ffm_model.setTest(write_path+'test_online.ffm')  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict(write_path+'model.out', write_path+'output.txt')