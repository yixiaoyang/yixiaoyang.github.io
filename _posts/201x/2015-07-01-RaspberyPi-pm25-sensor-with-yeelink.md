---
author: leon
comments: true
date: 2015-07-20 12:37:52+00:00
layout: post
math: true
title: '[Python] Raspberry Pi检测PM2.5并同步到Yeelink'
categories:
- 嵌入式
tags:
- 嵌入式
---

几年前买的树莓派几乎没动，相对于Beaglebone太弱了，不过Rpi社区比Beaglebone活跃很多。

1. PM2.5传感器

    女朋友喉咙不太舒服，所以买了一个Pm2.5的Sensor监测一下空气质量，经过UART封装后可以轻松的连接到MCU，一个大洋150左右。注意不要买粉尘传感器，粉尘传感器分辨率太低。

2. 构建Rpi的运行系统

    素不喜欢定制的Debian之类系统，必须Buildroot一路自行编译uboot、kernel和rootfs下载到SD卡。

3. 添加USB无线适配器驱动

    我使用RTL8192芯片的EDUP NP8508GS适配器，需要编译RTL8192系列驱动，并在系统启动时modprobe。

4. 设置wp_suplication

    主要在`/etc/wpa_supplicant.conf`文件中设置无限ssid和密码，网关和dns看需要设置。

5. Python读取PM2.5传感器数值

    读出的数据已经被转成数字信号了，可以直接拿来换算。代码如下：

```python
#! /usr/bin/env python  
# -*- coding: utf-8 -*-  

#
# email sent app
# by leon@2015-07-09
#
# curl -d "{\"value\":50}" -H "U-ApiKey: my_key" 'my_url'
#
import time
import json
import requests

# example: http://api.yeelink.net/v1.1/device/115088/sensor/144111/datapoints
class YeelinkHelper:
    url = 'http://api.yeelink.net/v1.1'
    key = ''
    hdr = ''
    dev = ''
    sensors = []

    def __init__(self,key,dev,sensors):
        self.key = key
        self.dev = dev
        self.sensors = sensors
        self.hdr = {'U-ApiKey':key,'content-type': 'application/json'}
    def up(self,vals):
        idx = 0
        utime=time.strftime("%Y-%m-%dT%H:%M:%S")
        for sen in self.sensors:
            if idx < len(vals):
                post_url=r'%s/device/%s/sensor/%s/datapoints' % (self.url, self.dev, sen)
                data={"timestamp":utime , "value": vals[idx]}
                res=requests.post(post_url,headers=self.hdr,data=json.dumps(data))
                idx += 1
                print("utime:%s, url:%s, status_code:%d" %(utime,post_url,res.status_code))
                print(self.hdr)
                print(json.dumps(data))
```

6.注册Yeelink账号并编写python数据上传脚本
yeelink的IOT（物联网）服务可添加简单的传感器数值，由平台保存数据并进行可视化。其上传格式可参见开发者API手册。上传部分代码如下：

```python
    #! /usr/bin/env python
    # -*- coding: utf-8 -*-  

    #
    # pms module detect
    # by leon@2015-07-09
    #
    #

    # -*- coding: utf-8 -*-  

    import serial
    from yeelink import YeelinkHelper

    # global enum
    DATALEN     = 10
    SBYTE       = 0xaa
    EBYTE       = 0xab
    HDR_IDX     = 0     #0xaa
    CMD_IDX     = 1
    PM25L_IDX   = 2
    PM25H_IDX   = 3
    PM10L_IDX   = 4
    PM10H_IDX   = 5
    R1_IDX      = 6
    R2_IDX      = 7
    CHK_IDX     = 8
    END_IDX     = 9     #0xab

    # yeelink
    YEELINK_API_KAY = 'my_key'
    YEELINK_DEV = '115167'
    # pm2.5, pm10, cpu_temp
    YEELINK_SENSORS = ['144020','144022','144483']

    class RPiDev:
        @staticmethod
        def get_cpu_temp():
            cpu_temp_file = open( "/sys/class/thermal/thermal_zone0/temp" )
            cpu_temp = cpu_temp_file.read()
            cpu_temp_file.close()
            return float(cpu_temp)/1000

    class PmsMod:
        def __init__(self,p,ydev):
            self.port = p
            self.idx = 0
            self.maxLen = DATALEN*2
            self.datas = [0]*self.maxLen
            self.ydev = ydev

        def getPm25(self,pos):
            offset = pos*DATALEN + PM25L_IDX
            return float(self.datas[offset]+256*self.datas[offset+1])/10;

        def getPm10(self,pos):
            offset = pos*DATALEN + PM10L_IDX
            return float(self.datas[offset]+256*self.datas[offset+1])/10;        

        def collect(self):
            # read firs start byte
            byte=ord(self.port.read())
            while byte != EBYTE:
                byte=ord(self.port.read())
                print(byte)
            self.idx = 0

            count = 0
            while True:
                byte = self.port.read()
                self.datas[self.idx] = ord(byte)
                self.idx += 1

                if self.idx == self.maxLen:
                    count+=1
                    # every 5*2=10s
                    if count >= 5:
                        pos = self.idx/DATALEN-1
                        pm25 = self.getPm25(pos)
                        pm10 = self.getPm10(pos)
                        cpu_temp = RPiDev.get_cpu_temp()
                        vals = [int(pm25),int(pm10),int(cpu_temp)]
                        self.ydev.up(vals)
                        print("pm25=%.1fug/m3, pm10=%.1fug/m3 tmp=%.1f'C" %(pm25,pm10,cpu_temp))
                        count = 0
                    self.idx = 0

    if __name__ == "__main__":
        ydev1 = YeelinkHelper(YEELINK_API_KAY,YEELINK_DEV,YEELINK_SENSORS)
        port = serial.Serial("/dev/ttyAMA0",baudrate=9600,bytesize=8,parity='N',stopbits=1,xonxoff=0,timeout=3.0)
        mod = PmsMod(port,ydev1)
        mod.collect()
        port.close()
```
