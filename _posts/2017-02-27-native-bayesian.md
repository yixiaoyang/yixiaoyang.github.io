---
author: leon
comments: true
date: 2017-02-27 10:19:00+00:00
layout: post
title: '[机器学习]朴素贝叶斯在机器学习中的使用'
categories:
- 机器学习
tags:
- 机器学习
---


### 朴素贝叶斯定理

事件A，事件B相互独立时 <img src="http://latex.codecogs.com/svg.latex?P(AB) = P(A)P(B)">

事件A，事件B相互不独立时，则需要计算条件概率P(A\|B)。定义 P(A\|B)为事件B已经发生的前提下，事件A发生的概率。叫做事件B发生下时间A的条件概率。其基本求解公式为：
<img src="http://latex.codecogs.com/svg.latex?P(AB) = P(B)P(A|B)">
同理<img src="http://latex.codecogs.com/svg.latex?P(AB) = P(A)P(B|A)">也是成立的，因此有朴素贝叶斯公式：
<img src="http://latex.codecogs.com/svg.latex?P(B|A)=\frac {P(A|B)P(B)} {P(A)}">

朴素贝叶斯在机器学习的任务中，每个实例`x`可由属性值的合取描述，而目标函数`f(x)`从某有限集合`V`中取值。学习器被提供一系列关于目标函数的训练样例，以及新实例（描述为属性值的元组）`x=<a1,a2…an>`，然后要求预测新实例的目标值（或分类）。贝叶斯方法的目标是在给定描述实例的属性值`<a1,a2…an>`下，的到最可能的目标值`Vmap`（MAP，为最大似然估计）。

<img src="http://latex.codecogs.com/svg.latex?V_{map} =\underset{v_j \epsilon V}{arg\ max} P(v_j|(a_1,a_2..a_n) = \underset{v_j \epsilon V}{arg\ max} \frac{P(a_1,a_2..a_n | v_j)P(v_j)}{P(a_1,a_2..a_n)}=\underset{v_j \epsilon V}{arg\ max}{P(a_1,a_2..a_n | v_j)P(v_j)}">

其中`P(a1, a2.. an)`为常数因此可以消除，只需分子最大化即可。又因为属性集`{a1, a2.. an}`中的属性分别独立，所以进一步可得到：

<img src="http://latex.codecogs.com/svg.latex?V_{map} =\underset{v_j \epsilon V}{arg\ max}{P(a_1,a_2..a_n | v_j)P(v_j)} = \underset{v_j \epsilon V}{arg\ max} P(v_j) \coprod_{i=1}^{n} {P(a_i|v_j)}">

### 贝叶斯实例：细胞状态分类

假设某切片细胞中正常`w1`和异常`w2`两类的先验概率分别为`p(w1) = 0.9, p(w2) = 0.1`。 现有一待识别的细胞状态`x`，其由类`条件概率密度`分布曲线查的`p(x|w1) = 0.2, p(x|w2) = 0.4`， 尝试对此细胞`x`进行分类。实际求`P(w1|x)和P(w2|x)`的概率大小。先求`P(x)`有:

<img src="http://latex.codecogs.com/svg.latex?P(x)=\sum_{i=1}^{n} {P(x|w_i)P(w_i)} = P(x|w_1)P(w_1) + P(x|w_2)P(w_2)">

求得`P(x) = 0.2*0.9 + 0.4*0.1 = 0.22`

根据贝叶斯公式有`P(w1|x) = P(x|w1)*P(w1)/P(x) = 0.2*0.9/0.22 = 0.818`， `P(w2|x) = 1-P(w1|x) = 0.182`，因此`P(w1|x)`的概率更大，可将x状态分类为`w1（正常）`。
<br>


### 贝叶斯实例：垃圾邮件分类

**样本**：1000封邮件，每个邮件被标记为垃圾邮件或者非垃圾邮件  
**分类目标**：给定第1001封邮件，确定它是垃圾邮件还是非垃圾邮件  
**方法**：朴素贝叶斯  
**类别c**：垃圾邮件c1，非垃圾邮件c2  
**词汇表**：统计1000封邮件中出现的所有单词，记单词数目为N，即形成词汇表。  
**向量化**：将每个样本si向量化：初始化N维向量xi，若词wj在si中出现，则xij=1,否则为0，从而得到1000个N维向量。  

运用贝叶斯定理有
<img src="http://latex.codecogs.com/svg.latex?P(c|x)= \frac {P(x|c)P(c)}{P(x)} ">，其中
<img src="http://latex.codecogs.com/svg.latex?P(x|c)= P(x_1,x_2..x_n|c)} = \prod_{i=1}^{n}{P(x_i|c)} ">，而
<img src="http://latex.codecogs.com/svg.latex?P(x)= \prod_{i=1}^{n}{P(x_i)">，带入即可求算。

`P(xi|ci)`表示在cj的分类下，第i个单词xi出现的概率。  
`P(xi)`表示所有样本中单词xi出现的概率。  
`P(cj)`表示邮件cj出现的概率。  
<br>

#### 问题1：遇到生词：拉普拉斯平滑
朴素贝叶斯方法有个致命的缺点就是对数据稀疏问题过于敏感。遇到生词的情况下P(x|c)等于，P(c|x)也为0。

为了解决这个问题，可以等效地扩大样本的数量，使得**未出现特征值赋予一个“小”的值而不是0**。这就是拉普拉斯平滑做的事情，将`p(x1|c1)= n1/n`平滑后就是：`p(x1|c1)= (n1+1) /(n+N)`。其中N为样本总数，这种加一的平滑也称加一平滑。  
<br>

#### 问题2：一个词在样本中出现多次，和一个词在样本中出现一次，形成的词向量相同
将“出现 | 不出现”的布尔值改为词频计数。
<br>

### 贝叶斯网络
把系统中涉及的随机变量，根据是否条件独立绘制在一个有向图中，就形成了贝叶斯网络。水有点深。相关的有马尔科夫模型以后再仔细学习。
<br>

### 参考
- [朴素贝叶斯](https://zh.wikipedia.org/zh-hans/贝叶斯定理)
- [从贝叶斯方法谈到贝叶斯网络](http://blog.csdn.net/v_july_v/article/details/40984699)
