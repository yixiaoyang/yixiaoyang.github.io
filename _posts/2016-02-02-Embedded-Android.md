---
author: leon
comments: true
date: 2016-02-02 11:21:25+00:00
layout: post
math: true
title: '[Android] Android驱动开发二三事' 
categories:
- Android
tags:
- Android
- C/C++++
---

Android在嵌入式开发中越来越常见。底层出了linux内核基础的东西外，加了很多中间层的东西。大概梳理一下会遇到的一些问题。

### Android架构解析

一图胜千言

![android架构图](http://elinux.org/images/c/c2/Android-system-architecture.jpg)

### Android驱动

4.x以上的系统大同小异，几乎都是先保证linux kernel层驱动 =》 native驱动测试 =》 hal层 =》 framework层 =》 接口层 =》 app层这种方向给上层调用的。legency的方法需要编写驱动so文件打包到app，而最新的hardware service方法则需要编写硬件服务。

具体来说，legency驱动方法可以不需要Android源代码单独进行编译，完成一个基本legency驱动需要：

1. 写LINUX驱动
2. 写LINUX应用测试程序
3. 写HAL驱动代码，调用linux native接口控制硬件。
4. 写JNI接口，用来包装第二步写的应用（要用NDK来编译）生成一个.SO文件，相当于CE下的DLL
5. 写JAVA程序,专门写一个类包含.SO文件，然后在JAVA里调用.SO里的函数。例子，可以看NDK里面的Sample文件夹，里面有一些例子

对于使用硬件服务方式编写驱动，好处是比较灵活给上层调用，硬件隔离，需要Android源码配合编译，一般编写方式如下：

1. linux 字符设备驱动的编写(kernel/drivers/char/chr_dev/)
2. linux 字符设备驱动的验证程序(openplatform/android/externl/chr_dev/)
3. 硬件抽象层(hal)子程序编写(android/hardware/libhardware/moudles/chr_dev/ 及android/hardware/libhardware/include/)
4. jni 程序编写(android/framework/base/services/jni/)
5. aidl 编写(android/framework/base/core/java/android/os/)
6. aidl 接口具体实现(android/framework/base/services/java/com/android/server/)
7. 字符设备文件权限设置 (android/system/core/rootdir/ueventd.rc)
8. 字符设备ko文件，自动加载设置(android/openplatform/project/common/kk_4.4_overlay/rootfs/sbin/init.rc)

之前写个的一个指纹模块驱动代码结构，可以看出相对于传统的linux驱动，android驱动涉及到android架构中的HAL层、framwork（jni、service、接口层），然后才是app进行硬件服务接口调用。

```bash

hal_jni_biovox2/                                       
├── cp.sh
├── development
│   └── apps
│       └── BiovoX2 指纹模块的native测试程序
│           ├── Android.mk
│           ├── biovox2.c
│           ├── biovox2.h
│           ├── code.h
│           ├── main_entry.c
│           ├── Makefile
│           ├── tty.c
│           ├── tty.h
│           └── X2Main.c
├── device
│   └── softwinner
│       └── astar-y3
│           └── init.sun8i.rc 删除蓝牙对ttyS1权限的设置部分
├── frameworks
│   └── base
│       ├── Android.mk
│       ├── core
│       │   └── java
│       │       └── android
│       │           └── os
│       │               └── IBiovoxService.aidl 指纹模块接口定义
│       └── services
│           ├── java
│           │   └── com
│           │       └── android
│           │           └── server
│           │               ├── BiovoxService.java 指纹模块硬件服务实现，面向APP层的主要实现
│           │               └── SystemServer.java
│           └── jni
│               ├── Android.mk
│               ├── com_android_server_BiovoxService.cpp 指纹模块硬件服务对hal层的调用
│               └── onload.cpp
├── hardware 硬件hal层的实现
│   └── libhardware
│       ├── Android.mk
│       ├── include
│       │   └── hardware
│       │       └── fpm
│       │           └── biovox2.h
│       └── modules
│           └── fpm
│               ├── Android.mk
│               └── biovox2.c
├── packages
│   └── experimental
│       └── FP_TEST
│           └── src
│               └── com
│                   └── example
│                       └── fp_test
│                           └── MainActivity.java APP层测试程序
├── README.md
├── system
│   └── core
│       └── rootdir
│           └── ueventd.rc 开启串口ttyS1对普通用户的读写权限
└── testDev
    ├── Android.mk
    └── testDev.c


```

### Adnroid书籍

1. 《Embedded Android》讲android内部机制比较详细，必读
2. 《Android内核揭秘》
