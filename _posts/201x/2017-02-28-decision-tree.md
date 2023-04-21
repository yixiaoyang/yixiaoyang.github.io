---
author: leon
comments: true
date: 2017-02-28 10:19:00+00:00
layout: post
math: true
title: '[机器学习]决策树中的信息增益度量'
categories:
- 机器学习
tags:
- 机器学习
---

### 决策树模型
机器学习中，决策树是一个预测模型；他代表的是对象属性与对象值之间的一种映射关系。树中每个节点表示某个对象，而每个分叉路径则代表的某个可能的属性值，而每个叶结点则对应从根节点到该叶节点所经历的路径所表示的对象的值。决策树仅有单一输出，若欲有复数输出，可以建立独立的决策树以处理不同输出。

### 属性选择度量方法

初始的数据集 相对于已经分类完毕的数据集更加混乱。`决策树的目标是希望找到一组特征，让数据集分类结果确定，而且最大程度靠近正确的答案`。决策树每使用一个特征，都更接近正确的分类，让数据集都变得更整齐一点。衡量数据的混乱程度，常用的有两个标准：一个是信息熵，另外一个是基尼指数。决策树的分割策略是：优先使用信息增益最大、信息增益比最大、基尼指数最小的特征

#### 信息增益（Infomation Gain）

> 热力学中的熵概念：一个系统的内能总数叫焓。其中不能做功的内能叫熵。而在信息论中，熵是接收的每条消息中包含的信息的平均量。
>
> If the terms information gain and entropy sound confusing, don’t worry. They’re meant
> to be confusing! When Claude Shannon wrote about information theory, John von
> Neumann told him to use the term entropy because people wouldn’t know what
> it meant.

首先，定义一下信息量。如果有一个样本S内存在多个事件X = {x1,x2..,xn}，每个事件的概率分布P = {p1, ..., pn}，则每个事件本身的讯息（自信息）为：

$$ I(x_i)=-log_{2}{p(x_i)} $$

例如英语有26个字母，假如每个字母在文章中出现次数平均的话，每个字母的讯息量为：
$$ I(x_i)=-log_{2}{\frac{1}{26}}=4.7 $$

为了计算熵（entropy），则需要将每个状态的信息量加上去即：
$$ H(X)=-\sum_{i=1}^{n}{p(x_i)}{log_2{p(x_i)}} $$

而一个特定的属性分割训练集而导致熵降低，就是信息增益（infomation gain）的概念。对于系统S当前状态下的熵为`H(X)`，在其特征X中引入新的特征Y进行分割后的熵为`H(X|Y)`。因此特征Y的信息增益为：
$$ IG(X|Y)=H(X)-H(X|Y)=H(X)-\sum_{y \epsilon Y}{\frac{|S_y|}{|S|}H(S_y)} $$

其中\|S\|表示样本总数目，\|Sy\|表示特征y在样本S中的数目。

python实现熵的计算：

```python
from math import log
def calcEntropy(datas):
    entries = len(datas)
    labels = {}
    entropy = 0.0
    for data in datas:
        label = data[-1]
        if label not in labels.keys():
            labels[label] = 1
        else:
            labels[label] += 1
    for label in labels:
        prob = float(labels[label])/entries
        entropy -= prob*log(prob, 2)
    return entropy

array1=[
    [1,1,'yes'],
    [1,1,'yes'],
    [1,0,'no'],
    [0,0,'no'],
    [0,1,'no'],
]
array2=[
    [1,1,'yes'],
    [1,1,'yes'],
    [1,0,'no'],
    [0,0,'no'],
    [0,1,'maybe'],
]

print calcEntropy(array1)
print calcEntropy(array2)
```

输出为：
```
% python CalcEntropy.py
0.970950594455
1.52192809489
```

可以看出array1的熵为0.97，当对其最后一组值增加一个‘maybe’的信息后，array2的信息更加`复杂`，此时array2的熵为1.52。

#### 基尼指数
（略）

### 构建决策树

#### ID3 算法构建决策树
ID3采用自顶向下的贪婪搜索遍历可能的决策空间，构造过程焦点问题是“使用哪一个属性在根节点进行分割”，而判断的标准可采用信息增益衡量，选出最大增益的属性后将样本分割，再对左右子节点进行递归遍历。

```
If all examples have the same label:
    – return a leaf with that label
Else if there are no features left to test:
    – return a leaf with the most common label
Else:
    – choose the feature F that maximises the information gain of S to be the next node using Equation
    – add a branch from the node for each possible value f in F
    – for each branch:
        ∗ calculate Sf by removing F from the set of features
        ∗ recursively call the algorithm with Sf , to compute the gain relative to the current set of examples
```

#### 举个栗子

|day|outlook|temperature|humidity|windy|play|
|-|-|-|-|-|-|
|D1|sunny|hot|high|FALSE|no|
|D2|sunny|hot|high|TRUE|no|
|D3|overcast|hot|high|FALSE|yes|
|D4|rainy|mild|high|FALSE|yes|
|D5|rainy|cool|normal|FALSE|yes|
|D6|rainy|cool|normal|TRUE|no|
|D7|overcast|cool|normal|TRUE|yes|
|D8|sunny|mild|high|FALSE|no|
|D9|sunny|cool|normal|FALSE|yes|
|D10|rainy|mild|normal|FALSE|yes|
|D11|sunny|mild|normal|TRUE|yes|
|D12|overcast|mild|high|TRUE|yes|
|D13|overcast|hot|normal|FALSE|yes|
|D14|rainy|mild|high|TRUE|no|

##### 第一步：各属性的增益计算
```
先计算最后的label列，yes有9个，no有5个：  
H(play) = −(5/14)*log(5/14,2)\−(9/14)*log(9/14,2) =  0.940  

对outlook属性，有5个sunny，4个overcast，5个rainy。
- outlook = sunny时，    5个，3个no，2个yes
- outlook = overcast时， 4个，0个no，4个yes
- outlook = rainy时，    5个，3个yes，2个no
H(play|outlook) = P(o=sunny)H(play|o=sunny) + P(o=overcast)H(play|o=overcast) + P(o=rainy)H(play|o=rainy)  
H(play|o=sunny) = −(2/5)*log(2/5,2) − (3/5)*log(3/5,2) = 0.971  
H(play|o=overcast) = 0  
H(play|o=rainy) = 0.971  
H(play|outlook) = 0.971*(5/14) + 0.971*(5/14) = 0.6936  
IG(play|outlook) = H(play)-H(play|outlook) = 0.940-0.6936 = 0.2464

同理可求算其他属性的增益：
IG(play|temperature) =  0.029
IG(play|humidity) = 0.152
IG(play|windy) = 0.048
```

因此信息增益最大的属性为‘outlook’， 决策树的根节点应选择‘outlook’属性。

##### 第二步：左子节点各属性增益计算

从第一步用‘outlook’属性分出了三个子节点分支，‘sunny’，‘overcast’和‘rainy’。先从左子节点开始计算。  
先从集合$D(outlook=sunny) = {D1,D2,D8,D9,D11}$分支开始计算。此分支信息量 $H(play|o=sunny) = 0.971$

|day|outlook|temperature|humidity|windy|play|
|-|-|-|-|-|-|
|D1|sunny|hot|high|FALSE|no|
|D2|sunny|hot|high|TRUE|no|
|D8|sunny|mild|high|FALSE|no|
|D9|sunny|cool|normal|FALSE|yes|
|D11|sunny|mild|normal|TRUE|yes|

```
H(play|humidity=high) = 0
H(play|humidity=normal) = 0
H(play|humidity) = 0

H(play|temperature=hot) = 0
H(play|temperature=mild) = -(1/2)*log(1/2,2)-(1/2)*log(1/2,2) = 1
H(play|temperature=cold) = 0
H(play|temperature) = (2/5)*0- (2/5)*1 - (1/5)*0 = 0.4

H(play|windy=Fasle) = -(1/3)log(1/3,2)-(2/3)log(2/3,2) = 0.9183
H(play|windy=True) = -(1/2)*log(1/2,2)-(1/2)*log(1/2,2) = 1
H(play|windy=Fasle) = (3/5)*0.9183 + (2/5)*1 = 0.951

IG(play|humidity) = H(play|o=sunny) - H(play|humidity) = 0.971 - 0 = 0.971
IG(play|temperature) = H(play|o=sunny) - H(play|temperature) = 0.971 - 0.4 = 0.57
IG(play|windy) = H(play|o=sunny) - H(play|windy) = 0.971 - 0.951 = 0.02
```
因此，此节点选择(humidity)作为判断属性。依次类推递归计算信息增量选择判断属性直到划分完毕（直到所有节点都是`纯`的，熵为0，划分结束）。


### ID3算法的问题和ID4.5算法

#### ID3的缺点
1. ID3算法针对属性值单一的节点，会直接将其作为分割属性，这样虽然使划分充分纯净，但是对分类毫无用处。
2. ID3算法没有考虑连续性值特征，如长度，密度都是连续值，无法在ID3上使用。
3. ID3采用信息增益大的特征优先建立决策树的节点。然而在相同条件下，取值比较多的特征比取值少的特征信息增益大。比如一个变量有2个值，各为1/2，另一个变量为3个值，各为1/3，其实他们都是完全不确定的变量，但是取3个值的比取2个值的信息增益大。
> 信息增益存在一个内在偏置，它偏袒具有较多值的属性。

4. ID3算法没有对缺失值的情况考虑。
5. ID3算法没有考虑过拟合的问题。

#### ID3算法的改进

ID4.5为ID3算法的升级版，主要改进是使用了信息增益比而不是信息增益作为属性选择的度量。

##### ID3算法改进1：连续特征离散化
连续特征离散化，即将连续性属性分段成大小不同的区间。而分段的依据是比较各个分段点的信息增益大小。

##### ID3算法改进2：剪枝

决策树过拟合通常是由训练样本中含有随机错误或者噪声。当叶子样本过少时也有可能存在巧合的规律，引起过拟合的风险。解决这个问题有两种思路：
1. 及早停止树的生长：在ID3完美分类训练数据之前剪枝(pruning)
2. 后剪枝：允许过拟合，然后在训练完后将其修剪。

决策树的剪枝往往通过极小化决策树整体损失函数来实现，其思想为，比较从树中删除子节点后总体的损失和不删除子节点的损失大小，如果剪枝后损失更小则剪去该节点。


##### ID3算法改进3：使用信息增益比消除信息增益的偏袒
信息增益比（Gain Ratio）引入“分裂信息（Split Information）”的项作为惩罚因子，对分裂属性的广度和均匀度进行衡量。
$$ SplitInfo(S,Y)=-\sum_{i=1}^{n}{\frac{|S_i|}{|S|}log_2{\frac{|S_i|}{|S|}}} $$  
其中属性Y将样本S分割成${S_1,S_2..S_n}$等n个子样本集。分裂信息计算实际上就是样本S关于属性Y的熵。

信息增益比则为：$$ IGR(S,Y)=\frac{IG(S,Y)}{SplitInfo(S,Y)} $$ 


### 决策树的优缺点
相对于其他数据挖掘算法，决策树在以下几个方面拥有优势：
- 决策树易于理解和实现.人们在通过解释后都有能力去理解决策树所表达的意义。
- 对于决策树，数据的准备往往是简单或者是不必要的.其他的技术往往要求先把数据一般化，比如去掉多余的或者空白的属性。
- 能够同时处理数据型和常规型属性。其他的技术往往要求数据属性的单一。
- 是一个白盒模型如果给定一个观察的模型，那么根据所产生的决策树很容易推出相应的逻辑表达式。
- 易于通过静态测试来对模型进行评测。表示有可能测量该模型的可信度。
- 在相对短的时间内能够对大型数据源做出可行且效果良好的结果。

缺点：
- 缺乏伸缩性：由于进行深度优先搜索，所以算法收内存大小限制难以处理大的数据集。
