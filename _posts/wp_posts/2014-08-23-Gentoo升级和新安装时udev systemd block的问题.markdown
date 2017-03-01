---
author: leon
comments: false
date: 2014-08-23 17:17:42+00:00
layout: post
slug: gentoo%e5%8d%87%e7%ba%a7%e5%92%8c%e6%96%b0%e5%ae%89%e8%a3%85%e6%97%b6udev-systemd-block%e7%9a%84%e9%97%ae%e9%a2%98
title: Gentoo升级和新安装时udev systemd block的问题
wordpress_id: 318
categories:
- Gentoo
tags:
- gentoo
---

好久没update gentoo了，工作一直稳定，上星期sync portage tree后安装一些软件包时出现udev和systemd block的问题:

```
[blocks B      ] sys-fs/udev ("sys-fs/udev" is blocking sys-apps/systemd-198-r4) 
[blocks B      ] sys-apps/systemd ("sys-apps/systemd" is blocking sys-fs/udev-198-r6) 
```

社区那些破事儿，好久没看了，bug报告在这里： [https://bugs.gentoo.org/show_bug.cgi?id=462750](https://bugs.gentoo.org/show_bug.cgi?id=462750)

原来udev和system项目合并了。
> “Linux设备命名系统Udev项目与启动和管理后台daemons和服务的systemd<项目宣布合并，并将统一版本号，例如下一个版本的 systemd将从45跳到184。开发者称，Udev设备管理是systemd不可分割的一部分，两个项目的合并将能最小化管理负担，最小化代码重复，解决核心系统中的循环构建依赖性。”
> Systemd has been made to now install udev with it and satisfy the udev virtual instead of systemd and udev being split from the same source and installed separately for systemd users (not sure if this will be permanent). 


关于systemd：[http://wiki.gentoo.org/wiki/Systemd ](http://wiki.gentoo.org/wiki/Systemd)
因为是block问题所以只能二选一了，systemd社区以前鄙视的要死现在因为其对系统可以全面掌控所以被gnome青睐，udev和systemd合并时迟早的事吧。我起初试图将udev mask，use systemd掉，然后将所有依赖udev的包删除重装。发现很多依赖还是没法彻斯解决还把linux-utility删掉引起系统不可使用的问题，OMG。只好重新用stage重新安装文件系统，相当于重装系统了但是不用配置很多东西。

意识到udev的依赖广泛我调整策略，mask掉systemd转而使用udev，哈哈，效果很好，重装软件后就不再出现这个麻烦的冲突了，就是阔怜的i3处理器编译起来龟速，kdebase加xorg大概用了三个小时。对于升级遇到相同问题的情况还是看现有安装包依赖程度，如果依赖这两个包的软件过多还涉及一些重要的系统包，还是不建议升级这玩意儿，否则最大的可能是你会被迫-C删除一些系统包解决其中的依赖问题，到头来再emerge各种系统包，相当于重装了一遍，实在是烦。
