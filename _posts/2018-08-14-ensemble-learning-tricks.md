---
author: leon
comments: true
date: 2018-08-14 12:35:00+00:00
layout: post
title: '[机器学习]集成学习常用模型和方法'
categories:
- 机器学习
tags:
- 机器学习
---

集成学习最基本的思想是构建多个分类器，用某种策略将多个结果集成，输出最终学习的结果。

## Voting Classifiers：多分类器角度

最容易理解的选举方法，接收多个分类器的结果进行投票，得分最高的结果作为最终输出。

## Bagging & Pasting：多重采样角度

`Bagging（bootstrap aggregating）`方法重点在于bootstrap抽样方法。`bootstrap（自助法）`是有放回的抽样方法，得到统计量的分布以及置信区间。

>  When sampling is performed with replacement, this method is called bagging1 (short for bootstrap aggregating). When sampling is performed without replacement, it is called pasting


sk-learn中的BaggingClassifier使用：

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), 
    n_estimators=500,
    max_samples=100, 
    bootstrap=True, 
    n_jobs=-1
)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```

> Decision trees are sensitive to the specific data on which they are trained. 
> If the training data is changed (e.g. a tree is trained on a subset of the 
> training data) the resulting decision tree can be quite different and in 
> turn the predictions can be quite different.

bootstrap方法抽样中大部分样本会多次被抽取出来输入到分类器，有一部分则从未被抽取，这部分样本一般称为`out-of-bag (oob) instances`，bagging外的样本因为从未被使用，可以用来做交叉验证。`BaggingClassifier`中的`oob_score_`就是这部分样本的验证分数。

## Random Forest

随机森林是Boosting方法的代表作，在Kaggle上经常看到用这个模型。

### 优点
- 在数据集上表现良好（随机boosting抽样，保证样本空间多样性，由于每一棵树的样本都不是全部的样本，相对不容易over-fitting。正因为这个原因，随机森林无需刻意剪枝）
- 它能够处理很高维度（feature很多）的数据，并且不用做特征选择
- 在训练完后，它能够给出哪些feature比较重要（有降维过程）
- 在创建随机森林的时候，对generlization error使用的是无偏估计（？）
- 训练速度快（子节点运用决策树的话计算熵或者信息增益相对更快）
- 在训练过程中，能够检测到feature间的互相影响（？）
- 容易并行优化（树结构模型，易并行分配计算任务）


### 算法描述

1. 数据集D，集成模型总数T，子空间维度d
2. 对于每一个子模型 t (t < T)，从D中随即有放回地抽样得到样本集$$D_t$$
3. 随机选择d个特征，由此降低$$D_t$$的维度到d
4. 以无剪枝的方式在$$D_t$$上构建树模型$$M_t$$
5. 重复2～4直到得到T个集成模型，对模型$$M_1,M_2..M_t$$进行结果集成，对于每个测试集由投票选择输出结果。

决策树模型对训练数据的变化非常敏感，所以Bagging方法十分有用，它增强了集成模型的样本多样性，还可以减少每颗树的计算时间。

>
> The difference between Random Forest algorithm and the decision tree algorithm is that in Random Forest, 
> the process es of finding the root node and splitting the feature nodes will run randomly.


## Boosting
Boosting为了增加训练集的多样性，采取了更复杂的抽样方法。目前比较流行的当属`Adaptive Boost`和`Gradient Boost`方法。

>  The general idea of most boosting methods is to train predictors sequentially, each trying to correct its prede‐cessor. 


### Adaptive Boost
采用加权的方法，对错分样本加重权值，。

> For example, to build an AdaBoost classifier, a first base classifier (such as a Decision Tree) is
> trained and used to make predictions on the training set. The relative weight of misclassified 
> training instances is then increased. A second classifier is trained using the updated weights 
> and again it makes predictions on the training set, weights are updated, and so on 

### Gradient Boost

Gradient Boost与传统的Boost的区别是，每一次的计算是为了减少上一次的残差(residual)，而为了消除残差，在残差减少的梯度(Gradient)方向上建立一个新的模型。每个新的模型的生成是为了使之前模型的残差往梯度方向减少，与传统Boost对正确、错误的样本进行加权有着很大的区别。


### FT


### Stacking


### 参考
- Hands-On Machine Learning with Scikit-Learn and TensorFlow
- 