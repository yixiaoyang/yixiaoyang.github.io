---
author: leon
comments: true
date: 2018-08-20 16:39:00+00:00
layout: post
math: true
title: '[机器学习]Gradient Boost方法深入解读'
categories:
- 机器学习
tags:
- 机器学习
---



`Boosting`方法有两个代表`Adaptive Boost`和`Gradient Boost`，在之前的笔记（文章在这里[[机器学习]集成学习常用模型和方法](https://yixiaoyang.github.io/articles/2018-08/ensemble-learning-tricks)）中有详细简单介绍`Adaptive Boost`算法细节，书中的证明看得云里雾里没有仔细钻下去。这篇笔记着重就`Gradient Boost`展开学习一下，并计划了解一下比较Kaggle上热门的`XGBoost`库。`Gradient Boost`算法直接读wiki和原论文[《Greedy Function Approximation: A gradient Boosting Machine》](http://docs.salford-systems.com/GreedyFuncApproxSS.pdf)原汁原味，很多书和ppt上直接截图此开山鼻祖论文中的算法描述。`XGboost`的实现原理直接看官方文档和陈天奇的ppt（这个ppt对提升树的介绍写的非常friendly）。

### Gradient Boost概览

`Gradient Boosting = Gradient Descent + Boosting`，Gradient Boost与传统的Boost的区别是，每一次的计算是为了减少上一次的残差(residual)，而为了消除残差，在残差减少的梯度(Gradient)方向上建立一个新的模型。每个新的模型的生成是为了使之前模型的残差往梯度方向减少，与传统Boost对正确、错误的样本进行加权有着很大的区别。 

#### steepest-descent
传统的`最速下降`（又称梯度下降法，steepest-descent）一种最简单的梯度优化器(`estimator`)，相当于每次沿着导数的方向走一小步，最终就能走到一个最优值附近。用数学的语言解释最速下降法，是利用目标函数的一阶`泰勒展开`（泰勒展开后面再做复习笔记）近似优化过程，求得搜索方向的方法。

##### 理解和推导
对$$f(x+v)$$在x处进行泰勒一阶展开，有

$$
f(x+v) = f(x) + a \nabla f(x)^Tv + o(a) 
$$

其中a为步长，o(a)高阶无穷小可忽略，因此有

$$
f(x) - f(x+v) \approx -\nabla f(x)^Tv
$$

即对v在下降方向的前提下（$$-\nabla f(x)^Tv < 0$$），$$-\nabla f(x)^Tv$$可以看做为f在x到x+v的下降量。

为了使下降量尽可能地大，自然可以想到$$v=-\nabla f(x)^T$$时，下降量最大。严格地说，由`科西不等式`可知

$$
|-\nabla f(x)^Tv| \le ||-\nabla f(x)^T|| \  ||v||
$$

当且仅当$$v=-\nabla f(x)^T$$时，等号成立。

#### Gradient Boost回归算法

查阅资料时发现`Gradient Boost`有两种不同的描述方法，一种是残差迭代优化的方法，如上面**Hands-On Machine Learning with Scikit-Learn and TensorFlow**一书中的例子，直接去拟合残差来构建下一个子模型。另外一种是梯度迭代的方法，使用梯度下降求解，每个子模型都在学习前面模型的梯度下降值，下面 **Greedy Function Approximation: A gradient Boosting Machine**论文中给出的Gradient Boost基础算法流程就是这种描述。

1. 初始化损失函数
$$
F_0(x) = arg\ min_{\rho} \sum_{1=1}^N {L(y_i, \rho)}
$$


1. 逐步构建子模型，对于每个模型（最多M个模型）$$F_m(x)$$

    (1). 将前m-1个子模型的组合模型代入损失函数，计算损失函数的在梯度方向上的向量。
    
    $$
    \hat y_i  = -\Bigl[{\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}} \Bigr] _{F(x)=F_{m-1}(x)}  ，(i=1,N)
    $$

    如常见的损失函数$$L=(y_i-F(x))^2$$的梯度向量求算，有$$\hat{y} = -2(y_i-F(x))$$.

    (2). 梯度向量已经求得，接下来需要在梯度向量的方向上构建新模型$$h(x_i, a_m)$$，新模型以梯度方向为目标。
    
    $$
    a_m=arg\ min_{a,\beta} \sum_{i=1}^N[\hat{y_i}-\beta h(x_i;a)]^2)
    $$
    
    令此式等于0，新模型即在梯度向量方向，可求得新模型的参数$$a$$。

    (3). 将新模型集成，重新计算损失函数。
    
    $$
    \rho_m = arg\ min_{\rho}\sum_{i=1}^NL(y_i, F_{m-1}(x_i)+\rho h(x_i;a_m))
    $$

    优化方向是损失函数最小，对此式求导，令等于0，得到新的子模型集成的权重参数$$\rho$$。

    (4). 子模型集成，重复M次。

    $$
    F_m(x) = F_{m-1}(x)+\rho_m h(x;a_m)
    $$

#### Gradient Boost分类算法

`Gradient Boost`分类算法思想和回归版本没有区别，主要是任务输出不一样。回归输出连续的值，而分类输出离散值，即需要获得一个概率分布去逼近真正的分布，导致无法直接输出类别去拟合输出的误差。解决这个问题主要方法有两种：1）使用指数损失函数，这种情况退化成`Adaptive Boost`方法（下节讨论这两种方法的联系）。2）使用对数损失函数，如cross-entropy。

### Adaptive boost 和 Gradient Boost

在分类问题上，当`Gradient Boost`使用指数损失函数时，等同于`Adaptive Boost`，因此可以认为`Adaptive Boost`是`Gradient Boost`的一种特例（AdaBoost提升错分数据点的权重，Gradient Boosting计算梯度，两种优化思想有区别）。`Adaptive Boost`方法中常见的损失函数形式：

$$
L(f) = \frac{1}{m} \sum_{1}^{m} e^{-y_i f(x_i)} = \frac{1}{m} \sum_{1}^{m} e^{-y_i \sum_{j=1}^{N} \rho h_j(x_i)}
$$

其他的后面再补充，**Machine learning:A Probabilistic Perspective**一书讲的真是妙。

### 参考
- Greedy Function Approximation: A gradient Boosting Machine
- https://xgboost.readthedocs.io/en/latest/tutorials/model.html
- [Slides:Introduction to Boosted Trees](http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
- Machine learning:A Probabilistic Perspective
