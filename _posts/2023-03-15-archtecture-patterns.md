---
author: leon
comments: true
date: 2023-02-11 21:47+00:00
layout: post
math: true
toc: true
title: '[架构] 常用架构模式'
categories:
- 架构
tags:
- 架构
- 设计模式
---

<!-- TOC -->

- [架构风格](#%E6%9E%B6%E6%9E%84%E9%A3%8E%E6%A0%BC)
    - [分层模式N-Tier](#%E5%88%86%E5%B1%82%E6%A8%A1%E5%BC%8Fn-tier)
    - [MVC架构](#mvc%E6%9E%B6%E6%9E%84)
    - [微服务架构](#%E5%BE%AE%E6%9C%8D%E5%8A%A1%E6%9E%B6%E6%9E%84)
    - [大计算架构](#%E5%A4%A7%E8%AE%A1%E7%AE%97%E6%9E%B6%E6%9E%84)
    - [大数据架构](#%E5%A4%A7%E6%95%B0%E6%8D%AE%E6%9E%B6%E6%9E%84)
        - [Lambda架构](#lambda%E6%9E%B6%E6%9E%84)
- [编程模式](#%E7%BC%96%E7%A8%8B%E6%A8%A1%E5%BC%8F)
    - [事件驱动模式](#%E4%BA%8B%E4%BB%B6%E9%A9%B1%E5%8A%A8%E6%A8%A1%E5%BC%8F)
    - [发布订阅模式](#%E5%8F%91%E5%B8%83%E8%AE%A2%E9%98%85%E6%A8%A1%E5%BC%8F)
    - [解释器模式](#%E8%A7%A3%E9%87%8A%E5%99%A8%E6%A8%A1%E5%BC%8F)
    - [管道过滤器模式](#%E7%AE%A1%E9%81%93%E8%BF%87%E6%BB%A4%E5%99%A8%E6%A8%A1%E5%BC%8F)
    - [黑板模式](#%E9%BB%91%E6%9D%BF%E6%A8%A1%E5%BC%8F)

<!-- /TOC -->

架构模式描述了特定领域内系统组织的惯用方式。
`
## 架构风格
### 分层模式(N-Tier)

![](/images/architecture/n-tier-logical.svg)

**正交性原则：** 层与层之间的关系应该是正交的
> 实现正交的考虑点：  
> （1）、消除重复（代码内聚）  
> （2）、分离关注点（接口抽象）  
> （3）、管理依赖：缩小依赖的范围和向稳定的方向依赖（单向依赖、最小依赖）  

**单一抽象层次原则（SLAP）:** 同一层的组件处于同一个抽象层次

> ![](/images/architecture/n-tier-layer-desc.webp)

**依赖：** 下层模块距离客户业务较远，变更较少，可以沉淀为更为通用的能力或者平台。

**数据：**只有中间层（一般是业务层）才能访问数据层。

**安全：** 使用应用防火墙(WAF，web application firewall )隔离前端和外部网络；每一层构建自己的子网域，形成安全域。


### MVC架构

### 微服务架构

![](/images/architecture/microservice-architecture.png)

### 大计算架构

![](/images/architecture/big-compute-logical.png)

### 大数据架构

![](/images/architecture/big-data-logical.svg)

回顾超大规模数据处理的重要技术以及它们产生的年代：

![](/images/architecture/bit-data-technology-timeserise.jpg)


#### Lambda架构

![](/images/architecture/lambda.png)


选用开源组件实现的一个范例视图：

![](/images/architecture/lambda2.png)


## 编程模式

### 事件驱动模式

### 发布订阅模式

### 解释器模式

### 管道过滤器模式

### 黑板模式


