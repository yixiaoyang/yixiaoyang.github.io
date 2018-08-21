---
author: leon
comments: true
date: 2018-08-20 16:39:00+00:00
layout: post
title: '[机器学习]Gradient Boost Machine深入解读'
categories:
- 机器学习
tags:
- 机器学习
---



`Boosting`方法有两个代表`Adaptive Boost`和`Gradient Boost`，在之前的笔记（文章在这里[[机器学习]集成学习常用模型和方法](https://yixiaoyang.github.io/articles/2018-08/ensemble-learning-tricks)）中有详细简单介绍`Adaptive Boost`算法细节，书中的证明看得云里雾里没有仔细钻下去。这篇笔记着重就`Gradient Boost`展开学习一下，并计划了解一下比较Kaggle上热门的`XGBoost`库。`Gradient Boost`算法直接读wiki和原论文[《Greedy Function Approximation: A gradient Boosting Machine》](http://docs.salford-systems.com/GreedyFuncApproxSS.pdf)原汁原味，很多书和ppt上直接截图此开山鼻祖论文中的算法描述。`XGboost`的实现原理直接看官方文档和陈天奇的ppt（这个ppt写的非常friendly）。




### Gradient Boost概览

`Gradient Boosting = Gradient Descent + Boosting`，Gradient Boost与传统的Boost的区别是，每一次的计算是为了减少上一次的残差(residual)，而为了消除残差，在残差减少的梯度(Gradient)方向上建立一个新的模型。每个新的模型的生成是为了使之前模型的残差往梯度方向减少，与传统Boost对正确、错误的样本进行加权有着很大的区别。 

**Hands-On Machine Learning with Scikit-Learn and TensorFlow**中手动实现了一个简单的Gradient Boosted Regression Trees (GBRT)例子，通过多次对错分集进行训练得到多个子模型，最后对各个子模型输出`求和`得到最终输出。

```python
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)
```

Now train a second DecisionTreeRegressor on the residual errors made by the first predictor:

```python
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)
```

Then we train a third regressor on the residual errors made by the second predictor:

```python
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)
```

Now we have an ensemble containing three trees. It can make predictions on a new instance simply by adding up the predictions of all the trees:

```python
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
```

### Gradient Boost解读


#### steepest-descent
传统的`最速下降`（steepest-descent）一种最简单的梯度优化器(estimator)。

**Greedy Function Approximation: A gradient Boosting Machine**论文中给出的Gradient Boost基础算法流程如下：

1. 初始化损失函数
$$
F_0(x) = arg min_{\rho} \sum_{1=1}^N {L(y_i, \rho)}
$$


2. 逐步构建子模型，对于每个模型（最多M个模型）$$F_m(x)$$

    $$
    \hat y_i  = -\Bigl[{\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}} \Bigr] _{F(x)=F_{m-1}(x)}  ，(i=1,N)
    $$


    $$
    a_m=argmin_{a,\beta} \sum_{i=1}^N[\hat{y_i}-\beta h(x_i;a)]^2)
    $$

    $$
    \rho_m = argmin_{\rho}\sum_{i=1}^NL(y_i, F_{m-1}(x_i)+\rho h(x_i;a_m))
    $$

    $$
    F_m(x) = F_{m-1}(x)+\rho_m h(x;a_m)
    $$




### 参考
- Greedy Function Approximation: A gradient Boosting Machine
- https://xgboost.readthedocs.io/en/latest/tutorials/model.html
- [Slides:Introduction to Boosted Trees](http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
