---
author: leon
comments: true
date: 2015-03-25 15:43:00+00:00
layout: post
math: true
title: '[Beaglebone] 添加外接RTC DS1307支持' 
categories:
- Beaglebone

tags:
- Beaglebone
- C/C++
- Linux
- 嵌入式
---

AM335X RTC的电源要求是1.8V，而市面上常用的纽扣电池一般是3.3v，因此Beaglebon上使用TPS65217X电源芯片的LDO转了一下。实际使用时这个芯片不是很稳定坏过两次，所以改成了若干个独立时钟电源芯片搭建，CPU的MPU、CORE供电分别为1.1v、1.1v即降频到800MHZ。如此一来，AM335X供电也就比较麻烦，可能需要额外的LDO转换，相比之下，使用外接单独的RTC芯片处理反而方便。

使用的外接RTC为DS1307，通过i2c-2接口挂在总线。硬件连接完毕后需要做两件事：

**1. 去除原AM335X RTC的驱动和检测**
如果不去除，uboot启动失败，kernel出现panic。对于uboot，找到CONFIG_SPL_AM33XX_ENABLE_RTC32K_OSC和CONFIG_POWER_TPS65217定义，注释掉; 对于kernel，在所有devicetree和overlay里找到tc@44e3e000 定义并注释掉即可。原RTC硬件定义：
```
    rtc@44e3e000 {
    compatible = "ti,da830-rtc";
    reg = <0x44e3e000 0x1000>;
    interrupts = <75
                76>;
    ti,hwmods = "rtc";
};
```

**2. 添加新器件DS1307驱动**

添加此器件驱动需要两步：

a). Device Tree添加硬件描述，将器件同RTC驱动关联 

```
/*
 * i2c0: Not exposed in the expansion headers
 * i2c1: pins P9 17,18 (and 24,26)
 * i2c2: pins P9 19,20 (and 21,22)
 * 
 * Pin assignments
 *
 * Module     Connector
 * SCL     -> P9.19
 * SDA     <- P9.20
 * SQW     <- NC
 *
 */
/dts-v1/;
/plugin/;

/{
	compatible = "ti,beaglebone", "ti,beaglebone-black";

	/* identification */
	part-number = "BB-BONE-RTC-01";
	version = "00A1";
	
	/* state the resources this cape uses */
	exclusive-use =
        /* the pin header uses */
        "P9.20",	/* i2c2_sda */
        "P9.19",	/* i2c2_scl */
        /* the hardware ip uses */
        "i2c2";
		
	fragment@0 {
		target = <&am33xx_pinmux>;
		__overlay__ {
			bb_i2c2_pins: pinmux_bb_i2c2_pins {
				pinctrl-single,pins = <
                    0x178 0x73  // spi0_d1.i2c2_sda,  SLEWCTRL_SLOW | IMPUT_PULLUP | MODE3
                    0x17c 0x73  // spi0_cs0.i2c2_scl, SLEWCTRL_SLOW | INPUT_PULLUP | MODE3
				>;
			};
		};
	};

	fragment@1 {
		target = <&i2c2>;

		__overlay__ {
			status = "okay";
			pinctrl-names = "default";
			pinctrl-0 = <&bb_i2c2_pins>;
			
			/* this is the configuration part */
			clock-frequency = <100000>;	
			
			/* shut up DTC warnings */
			#address-cells = <1>;
			#size-cells = <0>;

			rtc@68 {
				compatible = "dallas,ds1307";
				reg = <0x68>;
			};
		};
	};	
};
```

b). Kernel添加对应驱动：在内核menuconfig搜索DS1307就可找到。

重新编译uboot、内核进入系统后应该能够看到/dev/crt0设备文件描述符，可使用hwclock命令对其进行读写。再配合ntpd进程网络更新时间即可保持最新的系统时间。
把最新的时间写入RTC：`hwclock -w -f /dev/rtc0`
读取RTC时间：`hwclock -r -f /dev/rtc0`
