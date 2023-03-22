---
author: leon
comments: true
date: 2016-11-04 00:16:00+00:00
layout: post
math: true
title: '为GoldenDict添加生词本功能及Anki制卡工具'
categories:
- C++
- 工具
tags:
- 工具
C++
---


Windows上用欧陆词典，Linux上用Goldendict，奈何Goldendict没有生词本，所以下载源码加了个小模块进去。

新增功能：

1. 生词 添/删/分组/星标  
2. 生词导出，导出的文本可导入到欧陆词典。  

git： [https://github.com/yixiaoyang/goldendict](https://github.com/yixiaoyang/goldendict)

干的事情也蛮简单，效果如图所示，生词本导出后我用来做Anki卡片的（背单词）。

![golden-dict.png](http://cdn4.snapgram.co/images/2016/12/12/golden-dict.png)

哦，对了另外写了个Anki卡片制作脚本，功能：

1. 从bing.com下载单词相关图片
2. 从iciba.com下载单词例句、音标、语音
3. 从有道词典下载格林斯词典完整释义
4. 制卡

然后Anki卡片这样：

![anki.jpg](http://cdn3.snapgram.co/imgs/2016/12/12/anki.md.jpg)

git： [https://github.com/yixiaoyang/pyScripts/tree/master/anki-tools](https://github.com/yixiaoyang/pyScripts/tree/master/anki-tools)

说来惭愧，自从没打游戏后只能背单词度日 （Ex me？） =_=
