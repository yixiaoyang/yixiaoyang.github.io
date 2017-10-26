---
author: leon
comments: true
date: 2017-10-26 19:10:00+00:00
layout: post
title: '[Tensorflow]Kaggle Digit Recognizer代码（Tensorflow + CNN）重新整理'
categories:
- leetcode
tags:
- 机器学习
- tensorflow
- 神经网络
---

今天看了《Hands-On Machine Learning with Scikit-Learn & TensorFlow》CNN的章节，细节处理上更清晰了些。用LeNet5网络将kaggle上的[ Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) 问题的代码重新整理实现了一遍。以100行为一个patch，feed完所有训练数据后随机打乱数据再次迭代，迭代3500次左右用时332s，训练集（90%的train.csv数据）上的准确率0.959961，测试集（剩下10%的train.csv数据）上的准确率为 0.96619。迭代次数增多准确率会缓慢增长15分钟迭代测试集上的准确率大概能到0.991，差不多算这个模型上的极限了，有空再调参看看有没有改善空间。

模型结构：
```
LeNet-5 Architecture
layer   operation       feature-maps    kernel  stride  size     activation
in      input           1(gray image)   -       -       28*28    -
C1      convolution     16              5*5     1       28*28    relu
S2      avg pool        16              2*2     2       14*14    relu
C3      convolution     32              3*3     1       14*14    relu
S4      avg pool        32              2*2     2       7*7      relu
F5      full connected  -               -       -       256      relu
out     full connected  -               -       -       10       -
```


部分输出：
>...  
>epoch 3200, training accuracy 0.961914, validate accuracy 0.96881  
>epoch 3300, training accuracy 0.959961, validate accuracy 0.96619  
>training done  
>total training time:
>332.004180908s

```python
# coding: utf-8
#!/usr/bin/python2

import tensorflow as tf
import pandas as pd
import numpy as np
import time

class DigitsModelCNN(object):
    def __init__(self):
        self.train_input = tf.placeholder(tf.float32, shape=[None,784])
        self.train_out = tf.placeholder(tf.float32, shape=[None,10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.sess = tf.Session()

        # 21000 =》100*210
        self.batch_size = 100
        self.epochs = 210*16
        self.learn_rate = 5e-4

    '''
    @func       Computes a 2-D convolution given 4-D input and filter tensors.
    @param      input   4-D input tensor of shape [batch, in_height, in_width, in_channels]
                filter  4-D filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    @return
    '''
    def conv2d(self, input, filter, stride_w=1, stride_h=1):
        return tf.nn.conv2d(input, filter, strides=[1,stride_w,stride_h,1], padding='SAME')

    '''
    @func       Performs the max pooling on the input.
    @param      input   4-D Tensor with shape [batch, height, width, channels] and type tf.float32
                ksize   A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
                strides A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor
    @return
    '''
    def max_pool_2x2(self, input, stride_w=2, stride_h=2):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,stride_w,stride_h,1], padding="SAME")

    '''
    @func       outputs random values from a truncated normal distribution.
    '''
    def init_w(self,shape):
        # the standard deviation is 0.1
        value = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(value)

    '''
    @func       outputs random values as bias
    '''
    def init_b(self,shape):
        value = tf.constant(0.1, shape=shape)
        return tf.Variable(value)

    '''
    @note LeNet-5 Architecture
            layer   operation       feature-maps    kernel  stride  size     activation
            in      input           1(gray image)   -       -       28*28    -
            C1      convolution     16              5*5     1       28*28    relu
            S2      avg pool        16              2*2     2       14*14    relu
            C3      convolution     32              3*3     1       14*14    relu
            S4      avg pool        32              2*2     2       7*7      relu
            F5      full connected  -               -       -       256      relu
            out     full connected  -               -       -       10       -
    '''
    def build(self):
        self.train_input = tf.placeholder(tf.float32, shape=[None,784])

        self.input = tf.reshape(self.train_input, [-1, 28, 28, 1])
        self.f_c1 = self.init_w([5,5,1,16])
        self.b_c1 = self.init_b([16])
        self.c1 = tf.nn.relu(self.conv2d(self.input, self.f_c1) + self.b_c1)
        self.s2 = self.max_pool_2x2(self.c1)

        self.f_c3 = self.init_w([5,5,16,32])
        self.b_c3 = self.init_b([32])
        self.c3 = tf.nn.relu(self.conv2d(self.s2, self.f_c3) + self.b_c3)
        self.s4 = self.max_pool_2x2(self.c3)

        self.w_f5 = self.init_w([7*7*32, 256])
        self.b_f5 = self.init_b([256])
        self.x_f5 = tf.reshape(self.s4, [-1,7*7*32])
        self.f5 = tf.nn.relu(tf.matmul(self.x_f5, self.w_f5) + self.b_f5)

        # out@10
        self.f5_drop = tf.nn.dropout(self.f5, self.keep_prob)
        self.w_out = self.init_w([256,10])
        self.b_out = self.init_b([10])
        self.out = tf.nn.softmax(tf.matmul(self.f5_drop, self.w_out) + self.b_out)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.train_out))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss)

        predict = tf.equal(tf.argmax(self.out,1), tf.argmax(self.train_out,1))
        self.accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))


    def train(self, train_x, train_y, test_x, test_y, keep_prob=0.1):
        print("start training")

        self.sess.run(tf.global_variables_initializer())

        batch_start = 0
        batch_end = batch_start + self.batch_size

        print(self.train_input.shape)
        print(self.train_out.shape)

        for epoch in range(self.epochs):
            _, loss, prob = self.sess.run([self.optimizer, self.loss, self.out],feed_dict={
                self.train_input :  train_x[batch_start:batch_end],
                self.train_out:     train_y[batch_start:batch_end],
                self.keep_prob :    keep_prob
            })

            if epoch %100 == 0:
                train_accuracy = self.sess.run(self.accuracy, feed_dict={
                    self.train_input:   train_x[0:1024],
                    self.train_out:     train_y[0:1024],
                    self.keep_prob:     1.0
                })
                validate_accuracy = self.sess.run(self.accuracy, feed_dict={
                    self.train_input:   test_x,
                    self.train_out:     test_y,
                    self.keep_prob:     1.0
                })
                print("epoch %d, training accuracy %g, validate accuracy %g" % (epoch, train_accuracy, validate_accuracy))

            batch_start = batch_end
            batch_end = batch_start + self.batch_size
            if(batch_end > train_x.shape[0]):
                print("reset batch")
                batch_start = 0
                batch_end = batch_start + self.batch_size
                train_x, train_y = self.permutation(train_x, train_y)

        print("training done")

    def permutation(selfself, x, y):
        sequence = np.random.permutation(x.shape[0])
        return x[sequence], y[sequence]

    def info(self):
        print("c1,s2,c3,s4,c5 shape:")
        print(self.c1.shape)
        print(self.s2.shape)
        print(self.c3.shape)
        print(self.s4.shape)
        print(self.f5.shape)
        print('-'*16)
        print(train_x.shape)
        print(train_y.shape)

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def load_data(filename, train_data=True, split=0.9):
    data_frame = pd.read_csv(filename)
    # (42000, 785)
    print(data_frame.shape)

    train_data_len = data_frame.shape[0]
    train_data_split = int(train_data_len*split)
    print(train_data_split)

    train_x = data_frame.iloc[:train_data_split, 1:].values
    train_x = train_x.astype(np.float)
    train_x = np.multiply(train_x, 1.0/255.0)

    train_y = data_frame.iloc[:train_data_split, 0].values
    train_y = dense_to_one_hot(train_y,10)

    validate_x = data_frame.iloc[train_data_split:, 1:].values
    validate_x = validate_x.astype(np.float)
    validate_x = np.multiply(validate_x, 1.0/255.0)

    validate_y = data_frame.iloc[train_data_split:, 0].values
    validate_y = dense_to_one_hot(validate_y,10)

    print(train_x.shape)
    print(train_y.shape)
    print(validate_x.shape)
    print(validate_y.shape)
    return  train_x, train_y, validate_x, validate_y

train_x, train_y, validate_x, validate_y = load_data('./data/train.csv')

print(train_y.shape)
print(train_y[0:4,])

cnn = DigitsModelCNN()
cnn.build()
cnn.info()

time_start = time.time()
cnn.train(train_x, train_y, validate_x, validate_y)
time_end = time.time()
print("total training time:")
print(time_end-time_start)

```
