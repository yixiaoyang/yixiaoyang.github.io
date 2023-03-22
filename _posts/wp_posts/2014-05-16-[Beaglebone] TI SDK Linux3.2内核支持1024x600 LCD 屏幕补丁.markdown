---
author: leon
comments: true
date: 2014-05-16 01:37:52+00:00
layout: post
slug: beaglebone-ti-sdk-linux3-2%e5%86%85%e6%a0%b8%e6%94%af%e6%8c%811024x600%e5%b1%8f%e5%b9%95%e8%a1%a5%e4%b8%81
title: '[Beaglebone] TI SDK Linux3.2内核支持1024×600 LCD 屏幕补丁'
wordpress_id: 270
categories:
- Beaglebone

tags:
- Beaglebone
- C/C++
- Linux
- 嵌入式
---

之前在Beaglebone主板上使用chipsee 7寸800×480分辨率的LCD，准备使用另外一款1024×600的屏幕，因为bpp以及分辨率的改变，需要重新移植驱动。弄了两天，终于可以点亮屏幕正常显示。

linux版本：TI官方SDK ti-sdk-am335x-evm-06.00.00.00/board-support/linux-3.2.0-psp04.06.00.11

（1）LCD时钟及像素时钟设置

此处需要参考TI AM335X芯片SRM文档，Table 8-24. Per PLL Typical Frequencies(MHz) 和第8章PRCM模块和13章LCD Controller相关手册内容。对于Display PLL则需要弄清其时钟源和分频比例，以确保输出的LCD时钟在LCD的参考范围内。


[![am335-pll](http://cdn1.snapgram.co/imgs/2015/07/20/am335-pll.gif)](http://cdn1.snapgram.co/imgs/2015/07/20/am335-pll.gif)

图1 AM335X Display PLL硬件框图


AM335X LCD PLL的时钟源的选取不是唯一的，如上图所示，`LCD_CLK`并不一定要使用`Display PLL` 的CLKOUT，还可以使用`CORE_CLKOUTM5`或者`PER_CLKOUTM2`（固定192MHZ）作为时钟。此版内核使用的是系统时钟（CLKOUT）分频。当然，linux内核对时钟和Framebuffer的设置有了很方便的配置框架，很多硬件寄存器细节不需要自己编写了。

```c
setup_pin_mux(lcdc_pin_mux); 

if (conf_disp_pll(500000000)) {
  pr_info("Failed configure display PLL, not attempting to"
      "register LCDCn");
  return;
}

if (am33xx_register_lcdc(&TFC_S9700RTWV35TR_01B_pdata))
  pr_info("Failed to register LCDC devicen");
```

（2）色深参数设置。注意QT、Linux内核对16/24/32 bpp的支持。最开始使用32bpp参数，结果红蓝色 swapped，换成24bpp后恢复正常。为了不对framebuffer大改，就使用了24真彩色，也够用了

<pre>
14 
15 @@ -122,7 +122,7 @@ static struct lcd_ctrl_config lcd_cfg =
16     .ac_bias        = 255,
17     .ac_bias_intrpt     = 0,
18     .dma_burst_sz       = 16,
19 -   .bpp            = 32,
20 +   .bpp            = 24,
21     .fdd            = 0x80,
22     .tft_alt_mode       = 0,
23     .stn_565_mode       = 0,
24 @@ -137,7 +137,7 @@ static struct lcd_ctrl_config lcd_cfg =
25  struct da8xx_lcdc_platform_data TFC_S9700RTWV35TR_01B_pdata = {
26     .manu_name      = "ThreeFive",
27     .controller_data    = &lcd_cfg,
28 -   .type           = "TFC_S9700RTWV35TR_01B",
29 +   .type           = "INNOLUX_TN92",
30  };
31
</pre>


（3）frame buffer的扫描宽度、偏移设置，需要参照LCD数据手册，设置行、列同步扫描像素的参数，确保其在手册规定的范围内

<pre>
15 diff -Narup -X diff.ignore linux-3.2.0-psp04.06.00.11.org/drivers/video/da8xx-fb.c linux-3.2.0-psp04.06.00.11/drivers/video/da8xx-fb.c
716 --- linux-3.2.0-psp04.06.00.11.org/drivers/video/da8xx-fb.c 2013-06-25 21:38:14.000000000 +0000
717 +++ linux-3.2.0-psp04.06.00.11/drivers/video/da8xx-fb.c 2014-05-14 09:55:11.000033638 +0000
718 @@ -290,6 +290,33 @@ static struct da8xx_panel known_lcd_pane
719         .pxl_clk = 9000000,
720         .invert_pxl_clk = 0,
721     },
722 +  [4] = {
723 +#if 0
724 +    .name = "INNOLUX_TN92",
725 +    .width = 800, 
726 +    .height = 480, 
727 +    .hfp = 1, 
728 +    .hbp = 45,
729 +    .hsw = 48,
730 +    .vfp = 12,
731 +    .vbp = 20,
732 +    .vsw = 2, 
733 +    .pxl_clk = 25000000,
734 +#else
735 +   /* whole 1312*635 in parameter.sv, by xiaoyang, 12May14 */
736 +   .name = "INNOLUX_TN92",
737 +   .width = 1024,
738 +   .height = 600,
739 +   .hfp = 128,
740 +   .hbp = 100,
741 +   .hsw = 60,
742 +   .vfp = 12,
743 +   .vbp = 16,
744 +   .vsw = 7,
745 +   .pxl_clk = 50000000,
746 +#endif
747 +   .invert_pxl_clk = 0, 
748 +   },   
749  };
</pre>

由于我另外加上了chipsee的touchscreen驱动，暂时还用不上，可去掉。

打上/撤销补丁：

<pre>
cd kernel-source
patch -p1 <../kernel.patch
patch -R -p1 <../kernel.patch
</pre>


