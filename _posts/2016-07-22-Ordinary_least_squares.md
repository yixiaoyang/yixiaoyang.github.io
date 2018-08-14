---
author: leon
comments: true
date: 2016-07-22 17:00:00+00:00
layout: post
title: '普通最小二乘法的原理理解'
categories:
- 数据分析
- 机器学习
tags:
- 线性代数
---

## 普通最小二乘法的原理理解

### 目标

线性最小二乘拟合(linear least square fit)常用于数据拟合，以寻找数据之间的关系方程。

我们遇到的实际问题是在校准ADC仪器时，读取到两个输入变量：  
- ADC测量值序列x：即ADC的原始读值。  
- 万用表测量值序列y:在认为万用表是准确的情况下用来校准ADC数据，分析出ADC测量之和真实值之间的关系方程，这样就可以用ADC值计算出真实的测量值。

硬件上实现理论上x,y之间是简单的线性方程模型：y=ax+b，校准过程即求算能够近似预测y值的斜率a和截距b的过程。

### 最小二乘法原理

对于数据(xi, yi)，定义 `残差`为:
<img src="http://latex.codecogs.com/svg.latex? \varepsilon_i=(a x_{i} +b) - y_{i}">
残差的出现是数据测量误差引起，也可能是外部干扰引起，如果错误地估计了(a,b)则实际计算出来的残差将比较大，所以残差越小，似然函数方程越能真实地反映真实测量值。通用方法是使用最小化残差的平方和（即<img src="http://latex.codecogs.com/svg.latex?\min_{a,b}  \sum  \varepsilon_i^{2}">）来估计方程的准确性，其原因有：
1. 平方不在乎正负，符合我们的目标（使似然函数偏差尽可能小）
2. 平方给了残差一个加权方法，残差大的权重越大，但是这也使拟合方程更容易受你和方程影响。
3. 在残差服从均值0，方差 <img src="http://latex.codecogs.com/svg.latex?\sigma^2"> 的正态分布，且残差与x相互独立的假设下，最小二乘法估计结果与极大似然估计量（maximum likelihood estimator，使得似然函数最大化的估计）相同。
  > 确定拟合的标准应该被重视，并小心选择，较大误差的测量值应被赋予较小的权。并建立如下规则：被选择的参数，应该使算出的函数曲线与观测值之差的平方和最小。  
4. 最小二乘法计算相对简单

### 最小二乘法计算过程
1. 计算序列x和y的均值<img src="http://latex.codecogs.com/svg.latex?\overline x,\overline y">，x的方差Var(x)，x和y的协方差Cov(x,y)（covariance，衡量两个变量变化方向是否一致的统计量）。
2. 估计斜率，这是关键的一步，
<img src="http://latex.codecogs.com/svg.latex?a= \frac{Cov(x,y)}{Var(x)} =  \frac { \sum_{i=1}^{n} (\overline x-x_i)(\overline y-y_i)} { \sum_{i=1}^n (x_i-\overline x)^2}">  
3. 估计截距
<img src="http://latex.codecogs.com/svg.latex?b= \overline y - a \overline x">  

### 最小二乘法和证明
1. 设拟合去曲线为<img src="http://latex.codecogs.com/svg.latex?y=ax+b"> ，残差 <img src="http://latex.codecogs.com/svg.latex?d_i=(a x_{i} +b) - y_{i}">

2. 当 <img src="http://latex.codecogs.com/svg.latex?\sum_{i=1}^{n} d_i^2"> 最小时，拟合度最高

3. 令 <img src="http://latex.codecogs.com/svg.latex?D=\sum_{i=1}^{n} d_i^2=\sum_{i=1}^{n}(y_i-ax_i-b)^2">

  就函数D对a求偏导有
  <img src="http://latex.codecogs.com/svg.latex?\frac{\partial_D}{\partial_a} = \sum_{i=1}^{n} 2ax_i^2-2{x_i}{y_i}+2x_i{b}">

  就函数D对b求偏导有
  <img src="http://latex.codecogs.com/svg.latex?\frac{\partial_D}{\partial_b} = \sum_{i=1}^{n} 2b - 2y_i + 2a{x_i}">

  令 <img src="http://latex.codecogs.com/svg.latex?\frac{\partial_D}{\partial_a} = \frac{\partial_D}{\partial_b} = 0, n\overline{x}= \sum_{i=1}^n x_i,  n\overline{y}= \sum_{i=1}^n y_i">, 对满足偏导为0的参数对(a,b)，其描述的意义是y=ax+b此时完全拟合x,y序列。

  则对于a的偏导为0：
    <img src="http://latex.codecogs.com/svg.latex?\sum_{i=1}^{n} ax_i^2-{x_i}{y_i}+x_i{b} = 0">

  进一步推导得到`方程1`：
    <img src="http://latex.codecogs.com/svg.latex?a\sum_{i=1}^{n} x_i^2-\sum_{i=1}^{n}{x_i}{y_i}+nb \overline{x} = 0">

  则对于b的偏导为0，得到`方程2`，即截距求算公式：
  <img src="http://latex.codecogs.com/svg.latex?b-\overline{y}+a\overline{x}=0">

  将`方程2`代入`方程1`得到:

  <img src="http://latex.codecogs.com/svg.latex?a=\frac{ \sum_{i=1}^n{x_i}{y_i} - n \overline{x} \overline{y} }{ \sum_{i=1}^n x_i^2 - n\overline x^2 }">
  <img src="http://latex.codecogs.com/svg.latex?=\frac{ \sum_{i=1}^n x_i(y_i- \overline{y})} { \sum_{i=1}^n {x_i(x_i-\overline x)} }">
  <img src="http://latex.codecogs.com/svg.latex?=\frac { \sum_{i=1}^{n} (x_i-\overline x)(y_i-\overline y)} { \sum_{i=1}^n (x-\overline x)^2}">

  证明完成

### 最小二乘法和矩阵
典型的一次线性方程y = ax + b，写成矩阵式为：

<img src="http://latex.codecogs.com/svg.latex? min_{a,b} = \|
\begin{pmatrix}
  x_1 & 1\\
  x_2 & 1\\
  \vdots  & 1 \\
  x_n & 1
 \end{pmatrix}
\begin{pmatrix}
  a\\
  b
 \end{pmatrix}
-
\begin{pmatrix}
  y_1 \\
  y_2 \\
  \vdots \\
  y_n
 \end{pmatrix}
\|{_2} = min_a \|Xa-Y\|{_2}
">

即y离差平方和最小时，似然方程的拟合度越高。

### 回到实际应用问题

#### 原始数据
对DC部分测量电路的ADC作直流电流测量校准时，采集到ADC测量电流值X和万用表（串口远程获取）测量值序列Y，对其30次一组的取样作平均，得到如下一组数据：

```
% python regulation_mean.py
lenx leny:210 210
ADC测量值X(mA) 万用表测量值Y(mA)
0.0           76.7190062633
500.0         536.877983833
1000.0        986.017599667
1500.0        1436.35997433
2000.0        1889.30300333
2500.0        2344.180729
3000.0        2800.14835633
```
#### 最小二乘法计算

按照最小二乘法的算法：

```
% python main0.py
lenADC lenMM:7 7
avg:1500.000000 1438.515236
a=0.90629850  b=79.06749209
```
即拟合方程为 `y = 0.9062985 * x + 79.06749209`

python代码：

```python
#!/usr/bin/python
# -*- coding: utf-8 *-*

#
# 手动最小二乘法对校准系数进行拟合，得到校准方程
#
import os
import numpy as np

class Config:
    DATA_FILE="./regulate2.txt"


def data_import(file,delimiter):
    return np.genfromtxt(file,delimiter=delimiter)

if __name__ == "__main__":
    datafile = Config.DATA_FILE
    data = data_import(datafile,' ')

    mmVals = data[:,1]
    adcVals = data[:,0]
    print("lenADC lenMM:%d %d" %(len(adcVals),len(mmVals)))

    avgAdc = np.average(adcVals)
    avgMm = np.average(mmVals)
    print("avg:%f %f" % (avgAdc,avgMm))

    covXY = 0.0
    varX = 0.0
    for idx in range(len(adcVals)):
        # 计算协方差
        covXY = covXY + (adcVals[idx]-avgAdc)*(mmVals[idx]-avgMm)
        # 计算X方差
        varX = varX + (adcVals[idx]-avgAdc)*(adcVals[idx]-avgAdc)

    # 计算y=ax+b
    a= covXY/varX
    b = avgMm - avgAdc*a
    print ("a=%.8f  b=%0.8f" % (a,b))
```

#### numpy拟合计算

使用numpy拟合的结果如下：

```
% python main1.py
lenADC lenMM:7 7
[  0.9062985   79.06749209]
adc测量值         方程值          万用表值        残差          残差偏差千分比
0.00000000       79.06749209     76.71900626     2.34848582      0.00000000
500.00000000     532.21674009    536.87798383    -4.66124374     -9.32248748
1000.00000000    985.36598810    986.01759967    -0.65161157     -0.65161157
1500.00000000    1438.51523611   1436.35997433   2.15526178      1.43684119
2000.00000000    1891.66448411   1889.30300333   2.36148078      1.18074039
2500.00000000    2344.81373212   2344.18072900   0.63300312      0.25320125
3000.00000000    2797.96298013   2800.14835633   -2.18537620     -0.72845873
```
即拟合方程为 `y = 0.9062985 * x + 79.06749209`

python代码：

```python
#!/usr/bin/python
# -*- coding: utf-8 *-*

#
# 对校准系数进行拟合，得到校准方程
#
import os
import numpy as np
import matplotlib.pyplot as plot

class Config:
    DATA_FILE="./regulate2.txt"


def data_import(file,delimiter):
    return np.genfromtxt(file,delimiter=delimiter)

if __name__ == "__main__":
    datafile = Config.DATA_FILE
    data = data_import(datafile,' ')

    mmVals = data[:,1]
    adcVals = data[:,0]
    print("lenADC lenMM:%d %d" %(len(adcVals),len(mmVals)))

    plot.xlabel("adc")
    plot.ylabel("mm")
    plot.autoscale(tight=True)
    plot.grid()
    plot.scatter(adcVals,mmVals)
    plot.autoscale(tight=True)

    degs = [1]
    for deg in degs:
        fit = np.polyfit(adcVals,mmVals,deg)
        fnd = np.poly1d(fit)
        fx = np.linspace(0,adcVals[-1],10)
        plot.plot(fx,fnd(fx))
        print fit

    print ("%-12s\t %-12s\t %-12s\t %-12s\t %-12s" % ("adc测量值" , "方程值" , "万用表值" , "残差", "残差偏差千分比"))

    for idx in range(len(adcVals)):
        calc = fit[0]*adcVals[idx]+fit[1]
        diff = calc-mmVals[idx]
        diffPercent = 0
        if(adcVals[idx] != 0):
            diffPercent = (1000*diff)/adcVals[idx]
        print ("%-.08f\t %.08f\t %.08f\t %.08f\t %.08f" % (adcVals[idx] , calc , mmVals[idx] , diff, diffPercent))

    plot.show()
```

## 相关定义

### 协方差
在概率论和统计学中，协方差(covariance)可以用来衡量相关变量变化趋势是否相同。而方差是协方差的一种特殊情况，即当两个变量是相同的情况。假设我们有x和y序列，他们与其均值离差分别为：  
<img src="http://latex.codecogs.com/svg.latex?dx_i=x_i-\overline{x}">  
<img src="http://latex.codecogs.com/svg.latex?dy_i=y_i-\overline{y}">  
如果x，y变化方向一致，则他们与均值的离差应该有相同的正负号，如果讲二者的离差相乘，则当二者符号相同时，乘积为正数，所以这些乘积之和可以用来衡量两个序列变化是否一一致。因此，协方差的计算公式为：
<img src="http://latex.codecogs.com/svg.latex?Cov(x,y)= \frac{1}{n} \sum_{i=1}^n dx_i dy_i ">  

### 偏导
在一元函数中，我们已经知道导数就是函数的变化率。对于二元函数我们同样要研究它的“变化率”。然而，由于自变量多了一个，情况就要复杂的多。在xOy平面内，当动点由P(x0,y0)沿不同方向变化时，函数f(x,y)的变化快慢一般说来是不同的，因此就需要研究f(x,y)在(x0,y0)点处沿不同方向的变化率（来自xx百科）

几何意义：曲面上的每一点都有无穷多条切线，描述这种函数的导数相当困难。偏导数就是选择其中一条切线，并求出它的斜率。通常，最感兴趣的是垂直于y轴（平行于xOz平面）的切线，以及垂直于x轴（平行于yOz平面）的切线。一种求出这些切线的好办法是把其他变量视为常数。
