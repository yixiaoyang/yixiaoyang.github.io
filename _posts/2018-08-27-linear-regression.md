---
author: leon
comments: true
date: 2018-08-27 19:16:05+00:00
layout: post
title: '[机器学习]Linear Regression模型和优化方法'
categories:
- 机器学习
tags:
- 机器学习
---

线性回归在机器学习任务中非常常见且，模型相对简洁易实施，值得仔细学习。

## Linear Regression模型的基本任务

`Linear Regression`模型的基本预测方程:

$$
\tag{1} \hat{y} = {w_1}{x_1} + {w_2}{x_2}+ ... + {w_n}{x_n}
$$

用矩阵的形式表达为：

$$
\tag{2} h_{w}(X) = {w}^T \cdot X
$$

使用常见的MSE损失函数：

$$
\tag{3}  L(w) = MSE(X,h_{w}(X)) = \frac{1}{m} \sum_{i=1}^{m}(w^T \cdot X^{(i)} - y^{(i)})^2
$$

则`Linear Regression`模型的优化目标是找到使损失最小的参数 $$ \hat{w} $$。为解决线性回归问题，经典的机器学习算法（优化器）主要是`正则方程（最小二乘法）`和`梯度法`，其中`最小二乘法`在统计、微积分、线性代数、概率论等系统都有解释方法，`梯度法`则从基本的梯度下降法衍生出更实用的`批量梯度下降`、`随机梯度下降`和`小批量梯度下降`等经典算法。线性回归模型常见的是`多项式回归`，而对于实际问题中的非线性输出，传统线性模型无法做到因此引出`逻辑回归`，使用sigmoid等映射函数将输出映射到非线性空间去，除去映射函数其他步骤同传统线性模型并无区别。`逻辑回归`中最常见的经典模型必须是`Softmax回归`。基本上这些经典模型和算法吃透，线性回归模型


## Normal Equation

`Normal Equation`(正规方程)十分简洁，对于上面给定的描述，其形式表示为

$$
\tag{4} \hat{w} = (X^T \cdot X)^{-1} \cdot X^T \cdot y
$$

正规方程是最小二乘解的矩阵形式。

实际上 X^TX 不可逆的情况非常少就算 X^TX 真的是不可逆也无妨，可以对原始数据进行特征筛选或者是正则化即可。

### Normal Equation的推导和理解
从方程的角度看，方程（2）在现实中大部分情况是不可解的，所以进一步退而求其近似解 $$\hat{w}$$。

$$
\hat{w} = arg\ min_w L(w)= arg\ min_w {||y-Xw||}^2
$$

尝试展开L(w)有

$$
L(w) = {||y-Xw||}^2 \\
= {(y-Xw)}^T(y-Xw) \\
= y^Ty-(Xw)^Ty-y^TXw+(Xw)^T(Xw) \\
= y^Ty - 2w^TX^Ty + w^TX^TXw
$$

其中 $$X^Tw^Ty$$ 和 $$y^TXw$$ 得到的结果都是 1x1 的标量，对于标量 $$a$$， $$a^T = a$$，因此 $$w^TX^Ty = (w^TX^Ty)^T = y^TXw$$。

为最小化L(w)令其w的导数等于0有

$$
\frac{ \partial L(w) }{ \partial w} = \frac{\partial ({y^Ty - 2w^TX^Ty + w^TX^TXw})} {\partial w}
$$

其中

$$
\tag{a}  \frac{\partial (y^Ty)}{\partial w} = 0
$$

$$
\tag{b}  \frac{\partial (-2w^TX^Ty)}{\partial w} = -2X^Ty
$$

$$
\tag{c}  \frac{\partial (w^TX^TXw)}{\partial w} = 2X^TXw
$$

由(a)、(b)、(c)可得

$$
\frac{ \partial L(w) }{ \partial w} = -2X^Ty + 2X^TXw 
$$

令其等于0，可得

$$
\hat{w} =(X^TX)^{-1}X^Ty
$$

此处要求X矩阵是满秩矩阵。

> 矩阵求导可查询[wiki](https://en.wikipedia.org/wiki/Matrix_calculus)，另附参考文档：[matrix_rules.pdf](/attachments/matrix_rules.pdf)

## 梯度方法

梯度方法是机器学习中最常见的优化方法之一，从微积分的角度看对多元函数的参数求偏导，把求得的各个参数的偏导组成的向量就是该多元函数在各个参数上的`梯度`。在几何意义上，梯度的指向的方向是增长最快的方向，因此朝着梯度相反的方向可以快速找到最小值从而达到优化目的。

梯度方法相关的几个重要因素：
1. 步长：也即学习速率（learning rate），步长决定了每次梯度下降时参数变更的幅度。步长过打容易过冲无法找到最优点，步长过小结果更精确但训练将非常缓慢，

2. 特征：输入部分的相关特征选取可以优化计算速度，特征工程方面的内容后面再作学习（先挖坑）。另外在预处理输入是对特征归一化处理可以加快梯度优化速度。

3. 损失函数：一般使用MSE(L2损失)。因为他是凸函数，可以保证计算的是全局损失而不用担心可能算出一个局部损失。想对于MAE（L1损失），MSE的鲁棒性更差（对异常点敏感）。

### 批量梯度下降 (Batch Gradient Descent)

批量梯度下降法是梯度下降法的最基本形式，其特点是`每个参数每次迭代`都在`所有`训练数据上的计算梯度（当前参数在损失函数上的偏导）:

$$
\frac{ \partial L(w) }{ \partial w_j} = \frac{2}{m} \sum_{i=1}^{m}(w^T \cdot x^{(i)} - y^{(i)})x_j^{(i)}
$$

矩阵形式为：

$$
\nabla_{w} L(w) =  
\begin{pmatrix}
\frac{ \partial L(w) }{ \partial w_0}\\ 
\frac{ \partial L(w) }{ \partial w_1}\\ 
...\\
\frac{ \partial L(w) }{ \partial w_n}
\end{pmatrix}
=\frac{2}{m} X^T \cdot (X \cdot w - y)
$$

> **Hands-On Machine Learning with Scikit-Learn and TensorFlow** (Equation 4-6. Gradient vector of the cost function)
>
> Notice that this formula involves calculations over the full training set X, at each Gradient Descent step! This is why the algorithm is called Batch Gradient Descent: it uses the whole batch of training data at every step. As a result it is terribly slow on very large training sets (but we will see much faster Gradient Descent algorithms shortly). However, Gradient Descent scales well with the number of features; training a Linear Regression model when there are hundreds of thousands of features is much faster using Gradient Descent than using the Normal Equation.

参数更新方法：

$$
w^{j+1} = w^j - \eta \nabla_{w^j} L(w)
$$

### 随机梯度下降 (Stochastic Gradient Descent)

批量梯度下降法对`步长`的取值比较敏感，过大则精度不够，过小则收敛太慢，此外由于批量梯度下降法需要对整个训练集计算梯度，当数据较大时计算过程较慢。随机梯度下降方法则是将随机抽样概念引入基本梯度下降算法，采取动态步长和训练部分随机样本的一种策略，目的是优化精度且兼顾收敛速度。

> **Hands-On Machine Learning with Scikit-Learn and TensorFlow** (Page 117, Stochastic Gradient Descent)
> 
> Stochastic Gradient Descent  just picks a random instance in the training set at every step and computes the gradients based only on that single instance. Obviously this makes the algorithm much faster since it has very little data to manipulate at every iteration. It also makes it possible to train on huge training sets, since only one instance needs to be in memory at each iteration (SGD can be implemented as an out-of-core algorithm.
> 
> On the other hand, due to its stochastic (i.e., random) nature, this algorithm is much less regular than Batch Gradient Descent: instead of gently decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average. Over time it will end up very close to the minimum, **but once it gets there it will continue to bounce around, never settling down (see Figure 4-9). So once the algorithm stops, the final parameter values are good, but not optimal.**
 


### 小批量梯度下降

## 多项式回归

## 正则化的线性模型

### Ridge Regression（岭回归）


### Lasso回归

### Elastic Net（弹性网络）

## Logistic Regression（逻辑回归）

### 逻辑回归的本质

### Softmax


梯度下降特点:

    选择合适的学习速率α
    通过不断的迭代，找到θ0 ... θn, 使得J(θ)值最小

正规方程特点:

    不需要选择学习速率α，不需要n轮迭代
    只需要一个公式计算即可
    
## 参考
- http://mlwiki.org/index.php/Normal_Equation
- http://wiki.fast.ai/index.php/Gradient_Descent
- Hands-On Machine Learning with Scikit-Learn and TensorFlow
- https://www.cnblogs.com/maybe2030/p/5089753.html
 
## TODO
- 特征工程
