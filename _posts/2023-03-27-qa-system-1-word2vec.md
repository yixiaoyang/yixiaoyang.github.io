---
author: leon
comments: true
date: 2023-03-27 18:09+00:01
layout: post
math: true
toc: true
title: '[NLP] 对话系统系列技术文章-文本向量化'
categories:
- NLP
tags:
- 算法
- 深度学习
- 对话系统
- NLP
---


## 背景

现有机器学习系统无法直接处理文本数据，一般需要进行向量化，这个过程叫做Word Embedding（词嵌入）。

过去多年最常用的向量化方式是基于统计的离散方法表示，如独热编码（One-hot Representation）、词频-逆文本（TF-IDF）、词袋模型（Bag of Words）、N-gram等。

2013年，Word2Vec横空出世（相关论文见 #参考文档），自然语言处理领域各项任务效果均得到极大提升。自从Word2Vec这个神奇的算法模型出世以后，导致了一波嵌入（Embedding）热，基于句子、文档表达的word2vec、doc2vec算法，基于物品序列的item2vec算法，基于图模型的图嵌入技术相继诞生。Distributed Representation 逐渐流行起来。

小结一下主流的文本向量化方法。

**离散表示方法：**

- 独热编码（One-hot Representation）
- 词频-逆文本（TF-IDF）
- 词袋模型（Bag of Words）
- N-gram

**分布式表示方法：**

- Word2Vec
- Glove（Global Vectors）

## 分布式表示方法

![两种分布式表示处理方法对比](/images/nlp/word-distributed-representation.jpg)

### 基于推理的Word2Vec
基于推理的方法是使用神经网络，通常在mini-batch数据上进行学习，每次只看一部分学习数据，并反复更新权重。

Word2Vec有两种此训练模型： CBOW（Continuous Bag-of-Word）和Skip-Gram。

#### CBOW模型

![Continuous bag-of-word model](/images/nlp/CBOW.jpg)

#### Skip-Gram模型

![Skip-Gram model](/images/nlp/Skip-Gram.jpg)

#### Word2Vec高速化

##### Embedding层

##### 层次Softmax（hierarchical softmax）

##### 负采样（hierarchical softmax）

### 基于计数的Glove

基于计数的思路是使用整个语料库的统计数据（共现矩阵和PPMI等），经过一次降维处理（奇异值分解SVD、非负矩阵分解NMF、主成分分析PCA等常用方法）获得单词的分布式表示。


$J = \sum_{i,j=1}^V f(P_{ij})(\mathbf{w}_i^T\mathbf{w}j + b_i + b_j - \log P{ij})^2$

## 参考文档

- [Distributed Representations of Sentences and Documents]()
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781v3)
- [word2vec Parameter Learning Explained](https://www.researchgate.net/publication/268226652_word2vec_Parameter_Learning_Explained)
- [Glove算法原理及其简单理解](https://zhuanlan.zhihu.com/p/50946044)