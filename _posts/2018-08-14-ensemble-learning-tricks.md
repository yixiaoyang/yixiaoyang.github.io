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

bootstrap方法抽样中大部分样本会多次被抽取出来输入到分类器，有一部分则从未被抽取，这部分样本一般称为`out-of-bag (oob) instances`，bagging外的样本因为从未被使用，可以用来做交叉验证。`BaggingClassifier`中的`oob_score_`就是这部分样本的验证分数。

## Random Forest
 
### Ada Boost

### GB

### FT


### Stacking


### 参考
- Hands-On Machine Learning with Scikit-Learn and TensorFlow
- 
