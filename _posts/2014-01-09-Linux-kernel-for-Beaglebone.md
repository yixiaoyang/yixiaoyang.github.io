---
author: leon
comments: true
date: 2014-01-09 17:12:00+00:00
layout: post
title: '[Beaglebone] Linux kernel for Beaglebone' 
categories:
- 嵌入式
tags:
- 嵌入式
---


### Introduction

What is the most interesting part about the beaglebone family is it's expandability by using “Capes”, small add on boards that can stack, and allow easy access to the peripherals of the am3359 SoC. A plethora of capes have been produced for use for the Beaglebone White and most are compatible with the Black. Things are a bit complicated for capes that use the same pins as the onboard eMMC and the HDMI, to resolve this add-on capes have priority over the onboard peripherals.

The original beaglebone was shipped with a 3.2 kernel with a lot of patches and custom interfaces from Texas Instruments’s (TI) own kernel trees. Since this was based on a TI supported kernel, hardware compatibility was generally good and the capes worked. They were developed against it after all! 

#### Linux 3.2 or linux 3.8

> 之前在制作uboot时使用的linux 3.8内核由于device tree支持的问题，导致打印starting kernel启动内核失败。Linus Torvalds在2011年3月17日的ARM Linux邮件列表宣称“this whole ARM thing is a f*cking pain in the ass”，引发ARM Linux社区的地震，随后ARM社区进行了一系列的重大修正。在过去的ARM Linux中，arch/arm/plat-xxx和arch/arm/mach-xxx中充斥着大量的垃圾代码，相当多数的代码只是在描述板级细节，而 这些板级细节对于内核来讲，不过是垃圾，如板上的platform设备、resource、`i2c_board_info`、`spi_board_info` 以及各种硬件的`platform_data`。读者有兴趣可以统计下常见的s3c2410、s3c6410等板级目录，代码量在数万行。社区必须改变这种局面，于是PowerPC等其他体系架构下已经使用的Flattened Device Tree（FDT）进入ARM社区的视野。Device Tree是一种描述硬件的数据结构，它起源于 OpenFirmware (OF)。在Linux 2.6中，ARM架构的板极硬件细节过多地被硬编码在arch/arm/plat-xxx和arch/arm/mach-xxx，采用Device Tree后，许多硬件的细节可以直接透过它传递给Linux，而不再需要在kernel中进行大量的冗余编码。


#### Q&A 

**(1)ERROR:unrecognized/unsupported machine ID (r1 = 0x00000e05).**  

<pre>
Starting kernel ...
Uncompressing Linux... done, booting the kernel.
Error: unrecognized/unsupported machine ID (r1 = 0x00000e05).
Available machine support:
ID (hex)        NAME
ffffffff        Generic OMAP4 (Flattened Device Tree)
ffffffff        Generic AM33XX (Flattened Device Tree)
ffffffff        Generic OMAP3-GP (Flattened Device Tree)
ffffffff        Generic OMAP3 (Flattened Device Tree)
0000060a        OMAP3 Beagle Board
00000a9d        IGEP OMAP3 module
00000928        IGEP v2 board
00000ae7        OMAP4 Panda board
</pre>

引导内核后，如果出现这个错误的原因有很多:

- 一种是device tree描述文件am335x-bone.dtb没有正确加载引起;
- 也有可能是dtb文件加载位置被覆盖（ref：[[https://groups.google.com/forum/#!topic/beagleboard/WeAECofkl1k]]）;
- 还有可能是U-Boot版本过老对dtd没有支持。为了解决这个问题可以在编译内核时将dtd兼容选项编译进去，参考[[https://github.com/hvaibhav/am335x-linux/wiki/How-To-Use-Upstream-Tree|How-To-Use-Upstream-Tree]]。U-Boot 2012.10以后的版本都可以很好的支持dtd，所以在U-Boot中添加响应选项即可正确加载device tree表。我使用的uEnv.txt如下：

```
bootfile=uImage
devtree=/dtbs/am335x-bone.dtb
kloadaddr=0x80007fc0
dtboot=run mmcargs; fatload mmc ${mmcdev}:1 ${kloadaddr} ${bootfile} ; fatload mmc ${mmcdev}:1 ${fdtaddr} ${devtree} ; bootm ${kloadaddr} - ${fdtaddr}
uenvcmd=run dtboot
optargs=consoleblank=0
```

**(2)VFS: Cannot open root device "mmcblk0p2"**

- 普通列表项目检查内核fiiesystem下面对所挂载分区是否支持。如果不知道自己挂载的是什么分区，那么重新格式化储存卡，重新制作。支持的分区，一般ext2 ext3分区的兼容性较好，ext4在老一点的内核中不一定配置使用了。
- 检查kernel启动参数有bootwait

**(3)Ubuntu文件系统apache server后卡死**

其输出如下：

<pre>
[    4.542999] Freeing init memory: 236K
[    4.548034] Failed to execute /init.  Attempting defaults...
[    4.978881] init: ureadahead main process (844) terminated with status 5
[ OK ]ding cpufreq kernel modules...        
[ OK ]Freq Utilities: Setting ondemand CPUFreq governor...         * CPU0...        
udhcpd: Disabled. Edit /etc/default/udhcpd to enable it.
 * Starting web server apache2        AH00558: apache2: Could not reliably determine the server's fully qualified domain name, using 127.0.1.1. Set the 'ServerName' directive globally te
 * 
</pre>

### linux 3.8 from elinux.org

The modern BeagleBone kernels are Maintained by Koen Kooi and are available on the 3.8 branch at https://github.com/beagleboard/kernel/tree/3.8 . This repo contains a set of patches and a script which downloads a mainline kernel and then patches it appropriately. Exact steps for building it are in the README. 

`cp /opt/beaglebone/firmware/bin_am335x-pm-firmware.bin firmware/am335x-pm-firmware.bin`

### Reference
- [Linux device tree](http://elinux.org/Device_Tree) 
- [ARM Linux 3.x的设备树](http://blog.csdn.net/21cnbao/article/details/8457546) 
- [BeagleBone#Kernel](http://elinux.org/BeagleBone#Kernel|elinux.org) 
- [Device_Tree wiki](http://www.omappedia.org/wiki/Device_Tree)
- [Building uImage with DTB appended to the Kernel](https://github.com/hvaibhav/am335x-linux/wiki/How-To-Use-Upstream-Tree)
- [this whole ARM thing is a f*cking pain in the ass](https://lkml.org/lkml/2011/3/17/492)
