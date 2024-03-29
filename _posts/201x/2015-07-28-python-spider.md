---
author: leon
comments: true
date: 2015-07-28 17:16:00+00:00
layout: post
math: true
title: '[Python] 小型爬虫uSpider实现' 
categories:
- python
tags:
- python 
---

最近用python实现了一个小型爬虫，爬个博客没问题。
github地址： [https://github.com/yixiaoyang/pyScripts/tree/master/spider/uSpider](https://github.com/yixiaoyang/pyScripts/tree/master/spider/uSpider)

爬虫主要由三部分构成：
 
### 分析器（parser）

作为url生产者角色和doc的消费者角色，通过正则表达式解析下载文件中的链接产生url。由于解析速度较快，建议单线程。

### 下载器（downloader）

作为url消费者角色和doc的生产者角色。从网络下载文件时间不可预知，速度较慢，建议使用多进程

### 控制台（console）

报告当前的抓取情况，并维护url资源和doc资源。

1. url资源

    url资源由parser解析已经下载的文件内容后提取站点链接产生。如果url对应的文件已经存在，则表示此url已经下载过了，不再将其加入urls集合。

2. document资源

    downloader获取一条url后开始下载资源，下载完毕即在本地生成对应文件，每个站点文件为一个doc资源，parser将对每个未解析的doc文件资源进行链接提取。资源的同步使用两把资源锁。每个线程获取一条资源后需要将woker引用计数＋1表示仍有线程在对资源进行工作，当线程使用完资源后worker引用计数-1。当所有资源为空且资源计数为0（没有工作线程在引用资源）时，说明整个站点的解析和下载工作完成，此时线程可以此检测done状态退出，主进程随后join退出。

### 其他

1. 资源缓冲大小

    暂未设置资源缓冲大小，必要时需要加以限制防止在抓取大型站点时内存耗尽运行速率下降。

2. 相关站点抓取

    现有的程序仅广度优先遍历抓取一个站点内容，对其他站点的资源并没有抓取。后期对其他站点需要抓取一个深度的资源（直接引用资源）。

3. robots协议

    暂未考虑robots协议相关内容，是一只并不十分友好的spider。

4. 大名鼎鼎scrapy

5. 大型网站爬取过程中的问题
   - 去重问题，如果内存上的set不够用？
   - 集群化，如何进行分布式爬取
   - 爬取深度和权重，如何对资源的过滤
