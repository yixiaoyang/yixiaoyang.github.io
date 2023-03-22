---
author: leon
comments: true
date: 2020-04-07 17:40+00:00
layout: post
math: true
title: '[机器学习] 重新分析交通事故数据预测模型(三) - Tensorflow2.0上的RNN模型建立'
categories:
- 机器学习
tags:
- 机器学习
---

```python
import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, date, time

from sklearn.pipeline import Pipeline

import tensorflow as tf
#from tensorflow.contrib.layers import fully_connected
from tensorflow.keras import layers
import time

DATA_FILENAME = './2hours.csv'
acc_all = pd.read_csv(DATA_FILENAME)
acc_labels = acc_all['accident num']
acc = acc_all.copy()
acc.drop(['accident num'],axis=1,inplace=True)
# 时间列不需要，sequence序列可表示时间
acc.drop(['Time'],axis=1,inplace=True)
print(acc_all.shape)
print(acc.shape)

cat_attributes = acc.columns[[1]]
time_attributes = acc.columns[[0]]
num_attributes = acc.columns[[n for n in range(2,acc.shape[1])]]
print(cat_attributes)
print(time_attributes)
print(num_attributes)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.attribute_names].values

class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)
    def fit_transform(self, X, y=None):
        print("debug:fit_transform len(x)=%d"%(len(X)))
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)

class TimeAttribsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,time_from='01/01/2011 00:00'):
        self.from_dt = datetime.strptime(time_from, "%d/%m/%Y %H:%M")
    def fit(self,x, y=None):
        return self
    def __do_transform__(self, timeStr):
        #return timeStr
        day_time_splits = timeStr.split(' ')
        dt_format = "%d/%m/%Y %H:%M" if len(day_time_splits[0]) > len(day_time_splits[1]) else "%H:%M %d/%m/%Y"
        dt = datetime.strptime(timeStr, dt_format)
        dt_delta = dt-self.from_dt
        #hours = dt_delta.days*24+dt_delta.seconds/3600
        # 只关注小时时间
        hours= dt_delta.seconds/3600
        return hours/2
    def transform(self,x,y=None):
        # hours from 2000/1/1 0:00
        time_sequence = np.array([self.__do_transform__(str(val[0])) for val in x])
        return np.c_[time_sequence]    

# 时间类数据处理
time_pipeline = Pipeline([
    # 数据选择
    ('selector', DataFrameSelector(time_attributes)),
     # 数据选择
    ('time_transformer', TimeAttribsTransformer())
])

# 分类型数据处理num_attributes
cat_pipeline = Pipeline([    
    # 数据选择
    ('selector', DataFrameSelector(cat_attributes)),
    # 分类值独热编码
    ('label_binarizer', LabelBinarizerPipelineFriendly())
])

num_pipeline = Pipeline([
    # 数据选择
    ('selector', DataFrameSelector(num_attributes)),
    # 标准化：标准差标准化
    ('scaler', StandardScaler())
])

# pipeline集合
full_pipelines = FeatureUnion([
    ('num_pipeline',num_pipeline),
    ('cat_pipeline',cat_pipeline),
    #('time_pipeline',time_pipeline)
])

# 准备输入数据
acc_prepared = full_pipelines.fit_transform(acc)
print(acc_prepared.shape)
print(acc_labels.value_counts())
print(acc_prepared[0:4])

# 准备输出数据，分类任务，使用独热编码
print(acc_labels.value_counts())
label_pipeline = Pipeline([
    ('label_binarizer', LabelBinarizerPipelineFriendly())    
])
acc_labels_1hot = label_pipeline.fit_transform(acc_labels)
print(acc_labels_1hot.shape)

print(acc_labels.shape)
```

    (17532, 12)
    (17532, 10)
    Index(['holiday'], dtype='object')
    Index(['accident type'], dtype='object')
    Index(['precipitation', 'visibility', 'wind', 'wind direction', 'fog', 'rain',
           'sun rise', 'sun set'],
          dtype='object')
    debug:fit_transform len(x)=17532
    (17532, 9)
    0    16219
    1     1212
    2       98
    3        3
    Name: accident num, dtype: int64
    [[-0.06465394  0.2651457   1.15043477 -0.21958904 -0.20362638 -0.27402253
      -0.30910287 -0.31427953  1.        ]
     [-0.06465394  0.2651457   0.05587862 -0.21958904 -0.20362638 -0.27402253
      -0.30910287 -0.31427953  1.        ]
     [-0.06465394  0.2651457   0.71070873 -0.21958904 -0.20362638 -0.27402253
      -0.30910287 -0.31427953  1.        ]
     [-0.06465394  0.2651457   1.15043477 -0.21958904 -0.20362638 -0.27402253
       3.23516891 -0.31427953  1.        ]]
    0    16219
    1     1212
    2       98
    3        3
    Name: accident num, dtype: int64
    debug:fit_transform len(x)=17532
    (17532, 4)
    (17532,)
    


```python
from numpy import asarray
# split a univariate sequence into samples
def split_sequence(datax, datay, n_steps):
    X, y = list(), list()
    for i in range(len(datax)):
        end_ix = i + n_steps
        if end_ix > len(datax)-1:
            break
        seq_x, seq_y = datax[i:end_ix], datay[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)
```


```python
# define the rnn configs
n_inputs = 9
cell_type = 'LSTM'
n_batch = 32
n_neurons = 128
# 48*2 = 96hours
n_steps = 48
n_layers = 4
dropout_prob_in = 0.2
dropout_prob_out = 0.2

# 先使用较大学习速率，再使用较小学习完成收殓
learning_rate_init = 1e-5
learning_rate_decayed = 1e-6
accuracy_max = 0.95

n_split_train = 10000
```


```python
data_x, data_y = split_sequence(acc_prepared,acc_labels,n_steps)
print(data_x.shape)
print(data_y.shape)
print("n_inputs=",n_inputs)
```

    (17484, 48, 9)
    (17484,)
    n_inputs= 9
    


```python
train_x=data_x[:n_split_train,]
train_y=data_y[:n_split_train,]
test_x =data_x[n_split_train:,]
test_y =data_y[n_split_train:,]
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
```

    (10000, 48, 9)
    (10000,)
    (7484, 48, 9)
    (7484,)
    


```python
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM,SimpleRNN

def build_model(n_hidden=64, n_times=24, droup_out=0.2,print_summary=False):
    m = tf.keras.Sequential()
    m.add(LSTM(n_hidden, input_shape=(n_times, n_inputs)))
    m.add(Dropout(droup_out))
    m.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    m.add(Dropout(dropout_prob_out))
    m.add(Dense(1,activation='relu'))
    if(print_summary):
        m.summary()
    m.compile(loss='mse',
             optimizer='adam',
             metrics=['accuracy'])
    return m
```


```python
m1 = build_model(n_times=48)
history = m1.fit(train_x,
                train_y, 
                epochs=1, 
                validation_data=(test_x,test_y),
                batch_size=n_batch)
```
    Train on 10000 samples, validate on 7484 samples
    10000/10000 [=====] - 15s 1ms/sample - loss: 0.0593 - accuracy: 0.9469 - val_loss: 0.1317 - val_accuracy: 0.8948
    


```python
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribution = {
    "n_hidden":[16, 32, 64, 128],
    "n_times": [4,8,16,24,48,72,96,120],
    "droup_out": [0.05,0.1,0.2,0.3],
}

sk_model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn = build_model)
random_cv = RandomizedSearchCV(sk_model,
                               param_distribution,
                               n_iter = 10,
                               cv = 3,
                               n_jobs = 1)
random_cv.fit(train_x,
              train_y, 
              epochs=4, 
              validation_data=(test_x,test_y),
              batch_size=n_batch)
print(random_cv.best_params_)    # 返回最佳参数组合
print(random_cv.best_score_)     # 返回最佳评分 sklearn中回归问题默认mse
print(random_cv.best_estimator_) # 返回最佳模型
```

## 参数搜索

对模型影响最大的参数应该是时间长度n_times参数，决定了选取多少个小时作为输入序列数据。


```python
for time_width in [8,16,24,32,64,96]:
    x, y = split_sequence(acc_prepared,acc_labels,time_width)
    trainX=x[:n_split_train,]
    trainY=y[:n_split_train,]
    testX =x[n_split_train:,]
    testY =y[n_split_train:,]
    
    model = build_model(n_times=time_width)
    history = model.fit(trainX,
                trainY, 
                epochs=1, 
                validation_data=(testX,testY),
                batch_size=n_batch)
    print(time_width,history.history)

```
    Train on 10000 samples, validate on 7524 samples
    10000/10000 [=====] - 6s 575us/sample - loss: 0.0589 - accuracy: 0.9479 - val_loss: 0.1381 - val_accuracy: 0.8950
    8 {'loss': [0.05889062004705856], 'accuracy': [0.9479], 'val_loss': [0.13807493754044167], 'val_accuracy': 0.89500266]}
    Train on 10000 samples, validate on 7516 samples
    10000/10000 [=====] - 7s 726us/sample - loss: 0.0595 - accuracy: 0.9465 - val_loss: 0.1312 - val_accuracy: 0.8950
    16 {'loss': [0.05948311118425336], 'accuracy': [0.9465], 'val_loss': [0.13117971180052462], 'val_accuracy':[0.89502394]}
    Train on 10000 samples, validate on 7508 samples
    10000/10000 [=====] - 9s 895us/sample - loss: 0.0588 - accuracy: 0.9477 - val_loss: 0.1359 - val_accuracy: 0.8949
    24 {'loss': [0.05884406846314551], 'accuracy': [0.9477], 'val_loss': [0.1359180972137879], 'val_accuracy': 0.89491206]}
    Train on 10000 samples, validate on 7500 samples
    10000/10000 [=====] - 11s 1ms/sample - loss: 0.0588 - accuracy: 0.9477 - val_loss: 0.1345 - val_accuracy: 0.8949
    32 {'loss': [0.058750989521713926], 'accuracy': [0.9477], 'val_loss': [0.1344935974061794], 'val_accuracy': [0.89493334]}
    Train on 10000 samples, validate on 7468 samples
    10000/10000 [=====] - 18s 2ms/sample - loss: 0.0593 - accuracy: 0.9476 - val_loss: 0.1384 - val_accuracy: 0.8949
    64 {'loss': [0.05932865517017844], 'accuracy': [0.9476], 'val_loss': [0.13844861417379214], 'val_accuracy':[0.8948848]}
    Train on 10000 samples, validate on 7436 samples
    10000/10000 [=====] - 26s 3ms/sample - loss: 0.0594 - accuracy: 0.9475 - val_loss: 0.1387 - val_accuracy: 0.8948
    96 {'loss': [0.059361568695364804], 'accuracy': [0.9475], 'val_loss': [0.13865126840942582], 'val_accuracy': [0.89483595]}
    
