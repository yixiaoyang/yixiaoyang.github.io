---
author: leon
comments: true
date: 2018-08-10 20:12:00+00:00
layout: post
title: '[机器学习]KNN中的N和KD树结构'
categories:
- 机器学习
tags:
- 机器学习
---




## KNN工作原理

KNN方法简单且容易理解，给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的 k 个实例，这 k 个实例的多数属于某个类，就把该输入实例分为这个类。

**优点**：精度高、对异常值不敏感、无数据输入假定

**缺点**：计算复杂度高、空间复杂度高（KD树派上用场）

**适用数据范围**：数值型和标称型



## K值的选择

> **李航《统计学习方法》3.2.3**
>
> k 值的选择会对 k 近邻算法的结果产生重大的影响。
>
>如果选择较小的 k 值，就相当于用较小的邻域中的训练实例进行预测，“学习”的近似误差（approximation error）会>减小，只有与输入实例较近的（相似的）训练实例才会对预测结果起作用。但缺点是“学习”的估计误差（estimation error）会增大，预测结果会对近邻的实例点非常敏感。如果邻近的实例点恰巧是噪声，预测就会出错。换句话说，k 值的减小就意味着整体模型变得复杂，容易发生过拟合。
>
>如果选择较大的 k 值，就相当于用较大的邻域中的训练实例进行预测。其优点是可以减少学习的估计误差。但缺点是学习的近似误差会增大。这时与输入实例较远的（不相似的）训练实例也会对预测起作用，使预测发生错误。 k 值的增大就意味着整体的模型变得简单。



## KD-Tree

KD树(K-Dimensional Teee)和平衡二叉数非常相似，主要用于将数据在每个维度上进行相对均衡的划分（通过中位数划分当前维的空间，保证两边元素数目相近），常规增删查操作的平均算法复杂度为 $$O(log_n)$$。KNN算法中，在KD树上的最近邻点搜索比常规的线性扫描更快，从树根节点出发，在每一层的对应维度上选择其中较近的节点，这样便省去了另外一边节点的距离计算，查询的时间和节点深度相关。wikipedia上的参考算法实现如下，使用递归实现KD树构建：

```python
# source: https://en.wikipedia.org/wiki/K-d_tree

from collections import namedtuple
from operator import itemgetter
from pprint import pformat

class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))

def kdtree(point_list, depth=0):
    try:
        k = len(point_list[0]) # assumes all points have the same dimension
    except IndexError as e: # if not point_list:
        return None
    # Select axis based on depth so that axis cycles through all valid values
    axis = depth % k

    # Sort point list and choose median as pivot element
    point_list.sort(key=itemgetter(axis))
    median = len(point_list) // 2 # choose median

    # Create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kdtree(point_list[:median], depth + 1),
        right_child=kdtree(point_list[median + 1:], depth + 1)
    )

def main():
    """Example usage"""
    point_list = [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]
    tree = kdtree(point_list)
    print(tree)

if __name__ == '__main__':
    main()
```

输出

```
((7, 2),
 ((5, 4), ((2, 3), None, None), ((4, 7), None, None)),
 ((9, 6), ((8, 1), None, None), None))
```
