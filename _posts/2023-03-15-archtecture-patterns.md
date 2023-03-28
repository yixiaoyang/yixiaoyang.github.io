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

架构模式描述了特定领域内系统组织的惯用方式。

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


应用分层有两种方式：
- 水平分层：按照功能处理顺序划分应用，比如把系统分为web前端/中间服务/后台任务，这是面向业务深度的划分。
- 垂直分层：按照不同的业务类型划分应用，比如进销存系统可以划分为三个独立的应用，这是面向业务广度的划分，即业务垂直化方式。


### MVC架构

![](/images/architecture/mvc.png)

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

以之前设计的apprt为例，配置管理部分信息共享就是使用黑板模式，所有模块均可以往黑板（Redis的某些key上）进行数据擦写，也可以从所有的key上获取其他模块的配置项。

### Sidecar模式

![](/images/architecture/sidecar-enovy.webp)

Sidecar模式是一种将应用功能从应用本身剥离出来作为单独进程的方式。该模式允许我们向应用无侵入添加多种功能，避免了为满足第三方组件需求而向应用添加额外的配置代码。

Sidecar模式的好处：
- 通过将公用基础设施相关功能抽象到不同的层来降低微服务的代码复杂性
- 由于我们不需要在每个微服务中编写配置代码，因此减少了微服务架构中的代码重复
- 应用和底层平台之间实现了松耦合

以Envoy为例，有数据平面（Data Plane）和控制平面（Control Plane）两个概念：
- Data Plane：处理网格内服务间的通信，并完成服务发现、负载均衡、流量管理、健康检查等功能；数据平面的作用是处理网格内服务之间的通信，并负责实现服务发现、负载平衡、流量管理、健康检查等功能；
- Control Plane：管理和配置 Sidecar 来执行策略并收集遥测