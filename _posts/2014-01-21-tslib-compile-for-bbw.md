---
author: leon
comments: true
date: 2014-01-21 10:37:25+00:00
layout: post
title: '[Beaglebone] tslib for Beaglebone' 
categories:
- 嵌入式
tags:
- 嵌入式
---

编译QT4.8.5 for Beaglebone需要tslib库进行校准，此文档记录编译方法

1. 安装ti sdk

2. autogen

    ```bash
    sudo -s apt-get install libtool
    export CC=arm-linux-gnueabihf-gcc #不要export CC=arm-linux-gnueabihf-g++
    ./autogen.sh
    ```

3. config
如果直接运行configure 脚本相关的代码出现交叉编译错误: `undefined reference to 'rpl_malloc'`，这是由`ac_cv_func_malloc_0_nonnull`检查引起的，为了不让它检查，产生一个cache文件`daiq_tslib.cache`, 欺骗configure再执行

    ```
    $: echo "ac_cv_func_malloc_0_nonnull=yes" >daiq_tslib.cache`
    $: ./configure --host=arm-linux --cache-file=arm-linux.cache --prefix=$PWD/install --enable-inputapi=no ac_cv_func_malloc_0_nonnull=yes
    ```

4. 编译安装

`make && make install`


* [http://blog.csdn.net/yihui8/article/details/5753270](http://blog.csdn.net/yihui8/article/details/5753270)
