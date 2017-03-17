
kaggle上的入门问题，mnist手写识别，用cnn + tensorflow实现一遍，参考：https://www.kaggle.com/c/digit-recognizer


```python
import numpy as np
import pandas as pd
import tensorflow as tf

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
```

导入数据


```python
train_data = pd.read_csv("./data/train.csv")
images = train_data.iloc[:,1:].values
images = images.astype(np.float)
print(images.shape)
```

    (42000, 784)


归一化


```python
images = images/255.0
```


```python
labels = train_data.iloc[:,:1]
print (np.unique(labels))
print(labels.shape)
```

    [0 1 2 3 4 5 6 7 8 9]
    (42000, 1)


降维后独热编码


```python
'''
@func       one-hot encoding: convert label vector[?,1] to [?, 10]
            Convert class labels from scalars to one-hot vectors.
            
            0 => [1 0 0 0 0 0 0 0 0 0]
            1 => [0 1 0 0 0 0 0 0 0 0]
            ...
            9 => [0 0 0 0 0 0 0 0 0 1]

            The input to this transformer should be a matrix of integers, denoting the 
            values taken on by categorical (discrete) features. The output will be a 
            sparse matrix where each column corresponds to one possible value of one 
            feature. It is assumed that input features take on values in the 
            range [0, n_values).
'''
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
```


```python
# ravel will Return a contiguous flattened array
labels_flat = labels.values.ravel()
print (labels_flat.shape)
train_labels = dense_to_one_hot(labels_flat)
print (labels.shape)
```

    (42000,)
    (42000, 1)



```python
'''
@func       Computes a 2-D convolution given 4-D input and filter tensors.
@param      input   4-D input tensor of shape [batch, in_height, in_width, in_channels]
            filter  4-D filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
@return
@note       卷积运算
'''
def conv2d(input,filter):
    return tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')
```


```python
'''
@func       Performs the max pooling on the input.
@param      input   4-D Tensor with shape [batch, height, width, channels] and type tf.float32
            ksize   A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
            strides A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor
@return
@note       最大池化运算
'''
def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
```


```python
'''
@func       outputs random values from a truncated normal distribution.
'''
def init_w(shape):
    # the standard deviation is 0.1
    value = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(value)
```


```python
'''
@func       outputs random values as bias
'''
def init_b(shape):
    value = tf.constant(0.1, shape=shape)
    return tf.Variable(value)
```


```python
'''
@func       show image
'''
def show_image(label, image, width=28, height=28):
    plt.axis('off')
    plt.title(label,color="blue")
    plt.imshow(image, cmap=cm.binary)
```


```python
'''
@class  构建CNN网络
'''
session = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None,784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# x @28x28 + filter 32 features @5x5 => 32@24x24 => 32@12x12
w_conv1 = init_w([5,5,1,32])
b_conv1 = init_b([32])
# use RELU as active function
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 32@12x12 + filter 64 features @5x5 => 64@8x8 => 64@4x4
w_conv2 = init_w([5,5,32,64])
b_conv2 = init_b([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# MPL Densely Connected Layer
w_fc1 = init_w([7*7*64,1024])
b_fc1 = init_b([1024])
hpool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(hpool2_flat, w_fc1) + b_fc1)

print(x_image.get_shape())
print(h_conv1.get_shape())
print(h_pool1.get_shape())
print(h_conv2.get_shape())
print(h_pool2.get_shape())
print(hpool2_flat.get_shape())
print(h_fc1.get_shape())
```

    (?, 28, 28, 1)
    (?, 28, 28, 32)
    (?, 14, 14, 32)
    (?, 14, 14, 64)
    (?, 7, 7, 64)
    (?, 3136)
    (?, 1024)



```python
# Dropout removes some nodes from the network at each training stage. 
# Each of the nodes is either kept in the network with probability
# keep_prob or dropped with probability 1 - keep_prob. After the 
# training stage is over the nodes are returned to the NN with their
# original weights.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
w_fc2 = init_w([1024,10])
b_fc2 = init_w([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
y = tf.placeholder(tf.float32,[None,10])

print(y_conv.get_shape())
print(y.get_shape())

# optimizer and training evalution
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# prediction function
predict = tf.arg_max(y_conv,1)
```

    (?, 10)
    (?, 10)



```python
batch_n = 100
batch_idx = 0
epochs = 3
session.run(tf.global_variables_initializer())
for i in range(400):
    idx_start = i * batch_n
    idx_end = idx_start + batch_n
    train_x, train_y = images[idx_start:idx_end],train_labels[idx_start:idx_end]
    
    # test training accuracy
    if i % 100 == 99:
        feed_dict = {x: train_x, y: train_y, keep_prob: 1.0} 
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    
    # train batch
    feed_dict = {x: train_x, y: train_y, keep_prob: 1.0}
    train_step.run(feed_dict = feed_dict)
```

    step 99, training accuracy 0.9
    step 199, training accuracy 0.89
    step 299, training accuracy 0.9
    step 399, training accuracy 0.96



```python
test_data = pd.read_csv("./data/test.csv")
test_images = test_data.astype(np.float)
test_images /= 255.0
print(test_images.shape)
```

    (28000, 784)



```python
predicted_lables = np.zeros(test_images.shape[0])
for i in range(test_images.shape[0]/batch_n):
    idx_start = i * batch_n
    idx_end = idx_start + batch_n
    test_x = test_images[idx_start:idx_end]
    feed_dict = {x:test_x, keep_prob: 1.0}
    predicted_lables[idx_start:idx_end] = predict.eval(feed_dict=feed_dict)
print('predicted_lables({0})'.format(len(predicted_lables)))
```

    predicted_lables(28000)



```python
for i in range(14):
    plt.subplot(2,7,(i+1))
    img = test_data.iloc[i].as_matrix().reshape((28,28))
    plt.imshow(img, cmap='binary')
    plt.title(int(predicted_lables[i]),color="blue")
```


![png](output_20_0.png)



```python
print("done")
```

    done

