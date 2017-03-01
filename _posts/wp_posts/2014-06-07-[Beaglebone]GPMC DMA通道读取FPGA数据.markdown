---
author: leon
comments: false
date: 2014-06-07 12:16:49+00:00
layout: post
slug: beaglebonegpmc%e9%80%9a%e9%81%93%e8%af%bb%e5%8f%96fpga%e6%95%b0%e6%8d%ae
title: '[Beaglebone]GPMC DMA通道读取FPGA数据'
wordpress_id: 288
categories:
- Beaglebone
tags:
- Beaglebone
- C
- Linux
- 编程
---

为了实现FPGA和Beaglebone数据互通，且保证50Mbps的高速传输速率，我选择了AM3355x的GPMC（一般内存控制器）接口，当然数据搬运工作使用DMA能有效地节省CPU时间。小结一下，需要解决的问题有：



	
  1. FPGA的GPMC I/O驱动保证内核能够在无DMA情况下read/write数据。重点在于调试读写时序。

	
  2. FPGA的DMA驱动。第一步，保证内存到内存拷贝数据的DMA可用，保证DMA中断可以得到触发。第二部，保证FPGA IO到内存的DMA生效。


对与基本的读写功能，FPGA端做了一个简单的计数器，使用8bit数据线同D0-D7相连，同步模式，配置好GPMC模式，配合FPGA的GPMC时钟，以及 gpmc_timings，就能够按照预想的周期读写数据了。调试时示波器是少不了的，捕获读写周期才知道做的timing对不对。顺便说一下，论坛和参考博客里面硬写配置寄存器值得方式真是太挫了。我的测试timing如下，可以保证一次读一个byte：

    
    static struct gpmc_timings aplus_fpga_timings = {
       /* Minimum clock period for synchronous mode (in picoseconds) */
       .sync_clk = 1000 * GPMC_FPGA_SYNC_CLK_NS, /* 10ns */
       .cs_on = 0,
       .cs_rd_off = 2 * GPMC_FPGA_SYNC_CLK_NS,       /* Read deassertion time */
       .cs_wr_off = 2 * GPMC_FPGA_SYNC_CLK_NS,       /* Write deassertion time */
       /* ADV signal timings corresponding to GPMC_CONFIG3 */
       .adv_on = 0,            /* Assertion time */
       .adv_rd_off = 6 * GPMC_FPGA_SYNC_CLK_NS,       /* Read deassertion time */
       .adv_wr_off = 6 * GPMC_FPGA_SYNC_CLK_NS,       /* Write deassertion time */
       /* WE signals timings corresponding to GPMC_CONFIG4 */
       .we_on = 0 * GPMC_FPGA_SYNC_CLK_NS,        /* WE assertion time */
       .we_off = 6 * GPMC_FPGA_SYNC_CLK_NS,      /* WE deassertion time */
       /* OE signals timings corresponding to GPMC_CONFIG4 */
       .oe_on = 0 * GPMC_FPGA_SYNC_CLK_NS,        /* OE assertion time */
       .oe_off = 1 * GPMC_FPGA_SYNC_CLK_NS,      /* OE deassertion time */
       /* Access time and cycle time timings corresponding to GPMC_CONFIG5 */
       .page_burst_access = 1 * GPMC_FPGA_SYNC_CLK_NS,    /* Multiple access word delay */
       .access = 1 * GPMC_FPGA_SYNC_CLK_NS,       /* Start-cycle to first data valid delay */
       .rd_cycle = 2 * GPMC_FPGA_SYNC_CLK_NS,        /* Total read cycle time */
       .wr_cycle = 2 * GPMC_FPGA_SYNC_CLK_NS,        /* Total write cycle time */
       /* The following are only on OMAP3430 */
       .wr_access = 7 * GPMC_FPGA_SYNC_CLK_NS,        /* WRACCESSTIME */
       .wr_data_mux_bus = 3 * GPMC_FPGA_SYNC_CLK_NS,  /* WRDATAONADMUXBUS */
    };
    


第二个问题，我选择了一致性DMA，相对流式DMA的效率上的差别还真没仔细测试过，如果还需提高效率在测试看看。DMA以前只是概念上的理解，比较虚幻，写驱动前少不了补课。我先尝试使用流式DMA苦于没有正确读到数据，后来将TI的dma_test.c编译到内核测试ok，便使用一致性DMA尝试。中途发现驱动缓冲拷贝有问题导致可能读到GMPC IO数据但没有正确传回用户空间，饶了一圈，终于可以通过DMA读到硬件数据，剩下的就是GPMC burst模式下的时序微调了。



	
  1. 完成这种I/O数据通信的驱动，先需要做大量的硬件接口相关的功课。主要是通读并理解芯片的datasheet。

	
  2. 时序是重点，示波器不可少。

	
  3. 基于内核已有的类似的模块或者官方的Test Demo，调整程序和寄存器。

	
  4. 将问题分解，逐步解决，便可柳暗花明。


AM335x GPMC驱动和配置：[http://blog.chinaunix.net/uid-28818752-id-3655729.html](http://blog.chinaunix.net/uid-28818752-id-3655729.html)

读取数据为零的问题：[http://e2e.ti.com/support/arm/sitara_arm/f/791/t/204346.aspx](http://e2e.ti.com/support/arm/sitara_arm/f/791/t/204346.aspx)

读取数据不连续的问题，确认是D3管脚的问题，在Chipsee底板上同按键复用了，按键电容1uf导致高电平被拉低所以看起来D3位数据翻转有问题，

参考：[http://e2e.ti.com/support/arm/sitara_arm/f/791/p/345603/1209436.aspx#1209436](http://e2e.ti.com/support/arm/sitara_arm/f/791/p/345603/1209436.aspx#1209436)







[![](https://ff.duckduckgo.com/favicon.ico)](http://duckduckgo.com/?q=%3Cpre%20lang%3D%22LANGUAGE%22%20line%3D%221%22%20file%3D%22download.txt%22%20colla%3D%22%2B%22%3E%20%3C%2Fpre%3E&t=ff)[![](https://www.google.com/favicon.ico)](http://www.google.com/search?q=%3Cpre%20lang%3D%22LANGUAGE%22%20line%3D%221%22%20file%3D%22download.txt%22%20colla%3D%22%2B%22%3E%20%3C%2Fpre%3E)
