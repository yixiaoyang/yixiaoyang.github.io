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
\tag{3} MSE(X,h_{w}(X)) = \frac{1}{m} \sum_{i=1}^{m}(w^T \cdot X^{(i)} - y^{(i)})^2
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

$$
其中 X^Tw^Ty 和 y^TXw 得到的结果都是 1\times1 的标量，对于标量 a， a^T = a，因此 w^TX^Ty = (w^TX^Ty)^T = y^TXw。
$$

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

> 矩阵求导可查询[wiki](https://en.wikipedia.org/wiki/Matrix_calculus)，另附参考文档：[matrix_rules.pdf](/attachments/matrix_rules.pdf)

## 梯度方法

### 梯度下降法

### 批量梯度下降

### 随机梯度下降

### 小批量梯度下降

## 多项式回归

## 正则化的线性模型

### 岭回归

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
