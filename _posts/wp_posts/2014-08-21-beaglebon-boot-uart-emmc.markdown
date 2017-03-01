---
author: leon
comments: false
date: 2014-08-21 02:57:17+00:00
layout: post
slug: beaglebon-boot-uart-emmc
title: '[Beaglebone] Boot from UART and eMMC'
wordpress_id: 313
categories:
- Beaglebone
tags:
- Beaglebone
---

Hardware：Beagleble white A6
Kernel：linux-3.2-2013.01.01-psp06.00.00.00
U-boot：u-boot-2013.01.01-psp06.00.00.00

AM335X支持从MMC0/1接口启动，但是有个很不爽的限制（AM335x Version K，注意！只有最新的K版本文档才有这个限制的描述， Section 26.1.7.5.2）：

>**26.1.7.5.2 System Interconnection**
>
>Each interface has booting restrictions on which type of memory it supports:
>
>- MMC0 supports booting from the MMC/SD card cage and also supports booting from
>    eMMC/eSD/managed NAND memory devices with less than 4GB capacity.
>- MMC1 supports booting from eMMC/eSD/managed NAND memory device with 4GB capacity or
>    greater.
>
>The restriction is a result of many eMMC devices not being compliant with the eMMC v4.41 specification.
>If you have the need to boot from two different card cages, many MMC/SD cards will boot from MMC1, but
>for maximum compatibility only MMC0 should be used to boot from the card cage. Similarly for maximum
>compatibility, booting from eMMC/eSD/managed NAND should only be performed on MMC1.
>

手头上只有一个8G的eMMC卡，所以我试图从UART启动。
UART启动方式可参考文档：[`AM335x_U-Boot_Users_Guide`](http://processors.wiki.ti.com/index.php/AM335x_U-Boot_User%27s_Guide#UART).很多朋友问u-boot-spl.bin这个文件从哪里来，这个文件其实在uboot编译过程中就会产生，主要作用时spl初始化（板级基本初始化）。一个方法时可以find查找一下，二是可以看看uboot源码的Makefile就知道了，在编译文件目录的spl目录下面。

使用minicom上电，第一次用xmodem传输uboot-spl.bin,第二次传输uboot.img，稍等片刻即可进入uboot模式，然后扫描SD卡、eMMC、网络启动都不在话下了。
