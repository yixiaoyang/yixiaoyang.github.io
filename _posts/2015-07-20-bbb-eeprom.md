---
author: leon
comments: true
date: 2015-07-20 12:37:52+00:00
layout: post
title: '[Beaglebone] BBB eeprom的擦写' 
categories:
- 嵌入式
tags:
- 嵌入式
---


###EEPROM on BBB

半年前使用BBW（White版本）将eeprom物理损坏或者内容导致无法启动，原因暂不确定，现在使用正常的BBB板用I2C将eeprom重新连接起来进行擦写。

> BBB SRM文档的说明  
> A single 4KB EEPROM is provided on I2C0 that holds the board information. This information includes board name, serial number, and revision information. This is the not the same as the one used on the original BeagleBone. The device was changed for cost reduction reasons. It has a test point to allow the device to be programmed and otherwise to provide write protection when not grounded.
> ..


> The EEPROM used is the same one as is used on the BeagleBone and the eagleBone Black, a CAT24C256. The **CAT24C256** is a 256 kb Serial CMOS EEPROM, internally organized as 32,768 words of 8 bits each. It features a 64−byte page write buffer and supports the Standard (100 kHz), Fast (400 kHz) and Fast−Plus (1 MHz) I2 C protocol.
 

The BeagleBone is equipped with a single 32Kbit(4KB) 24LC32AT-I/OT EEPROM to allow the SW to identify the board. Table 7 below defined the contents of the EERPOM.

| NAME                | Size(Bytes) | Contents |
|---------------------|-------------|----------|
|Header               |4            | 0xAA, 0x55, 0x33, EE|
|Board Name           |8            | Name for board in ASCII: A335BNLT|
|Size(bytes)          |4            | Hardware version code for board in ASCII: 00A3 for Rev A3, 00A4 for Rev A4, 00A5 for Rev A5, 00A6 for Rev A6,00B0 for Rev B, and 00C0 for Rev C.|
|Serial Number        |12           | Serial number of the board. This is a 12 character string which is:`WWYY4P16nnnn` where: WW = 2 digit week of the year of production YY = 2 digit year of production BBBK = BeagleBone Black nnnn = incrementing board number|
|Configuration Option |32           | Codes to show the configuration setup on this board.All FF |
|RSVD                 |6            | FF FF FF FF FF FF|
|RSVD                 |6            | FF FF FF FF FF FF|
|RSVD                 |6            | FF FF FF FF FF FF|
|Available            |4018         | FF FF FF FF FF FF|


从BBW读出来数据如下：  

<pre>
U-Boot SPL 2013.01.01 (Jul 20 2015 - 11:39:45)
Initalize the board header
header magic:ee3355aa
header name:A335BONE00A64212BB000394
header version:00A64212BB000394
aa 55 33 ee 41 33 33 35 
42 4f 4e 45 30 30 41 36 
34 32 31 32 42 42 30 30 
30 33 39 34 00 00 00 00 
00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 
00 00 00 00 ff ff ff ff 
ff ff ff ff ff ff ff ff 
ff ff ff ff ff ff ff ff 
</pre>

###Device Address

> Address line A2 is always tied high. **This sets the allowable address range for the expansion cards to 0x54 to 0x57**. All other I2C addresses can be used by the user in the design of their capes. But, these addresses must not be used other than for the board EEPROM information. This also allows for the inclusion of EEPROM devices on the cape if needed without interfering with this EEPROM. It requires that A2 be grounded on the EEPROM not used for cape identification.

> This particular chip has 3 pins used for chip select addressing. However, those 3 pins alone don't make up the address. They are part of a hard coded binary prefix of '1010'. So, if you ground the three pins, the address really becomes '1010000', which is address 0x50 in hex. It will be important to know this address later. The chip also has a write protect pin. Because I want to write to it, I need to connect that to ground according to the datasheet. Connect up the chip Vss and Vdd and that covers all 8 pins of the chip. Pretty simple.

我使用Raspberry Pi作为烧写器，配置内核添加“i2c-dev,i2c-bcm2708”两个模块：`CONFIG_I2C_CHARDEV=M`和`CONFIG_I2C_BCM2708=M`，并将`I2C_BCM2708_BAUDRATE`设置成200k速率。

###I2C Tools
使用buildroot工具配置`BR2_PACKAGE_I2C_TOOLS=y`，添加i2cget/i2cset等工具支持.

```
# modprobe i2c_bcm2708
# modprobe i2c_dev
# ls /dev/i2c*
/dev/i2c-1
# lsmod
Module                  Size  Used by    Not tainted
i2c_dev                 4536  0 
i2c_bcm2708             4119  0 
i2c_core               16380  2 i2c_dev,i2c_bcm2708
# i2cdetect -y 1
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
	 00:          -- -- -- -- -- -- -- -- -- -- -- -- -- 
	 10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	 20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	 30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	 40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	 50: 50 -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	 60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	 70: -- -- -- -- -- -- -- --                         
# 

# i2cdump -y 1 0x50
No size specified (using byte-data access)
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f    0123456789abcdef
	 00: ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff    ................
	 10: ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff    ................
	 20: ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff    ................
	 30: ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff    ................
	 40: ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff    ................
	 50: ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff    ................
	 60: ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff    ................
	 70: ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff    ................
# 

# header
# i2cset -y 1 0x50 0 0xAA
# i2cset -y 1 0x50 1 0x55
# i2cset -y 1 0x50 2 0x33
# i2cset -y 1 0x50 3 0xEE
```

