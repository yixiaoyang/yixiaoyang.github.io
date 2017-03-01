---
author: leon
comments: true
date: 2014-05-07 11:37:00+00:00
layout: post
title: '[Beaglebone] U-boot and Buildroot for Beaglebone' 
categories:
- Beaglebone
---

### U-Boot for BeagleBone

#### Host Machine

<pre>
Ubuntu 12.04 LTS Intel® Core™ i3-3220 CPU @ 3.30GHz × 4
$ uname -a
Linux xiaoyang-desktop 3.2.0-57-generic #87-Ubuntu SMP Tue Nov 12 21:35:10 UTC 2013 x86_64 x86_64 x86_64 
GNU/Linux</code>
</pre>

### Buildroot for Beaglebone

#### Waht's BuildRoot
Buildroot is a set of Makefiles and patches that makes it easy to generate a complete embedded Linux system. 
Buildroot can generate any or all of a cross-compilation toolchain,a root filesystem, a kernel image and a 
bootloader image. Buildroot is useful mainly for people working with small or embedded systems, using various 
CPU architectures (x86, ARM, MIPS, PowerPC, etc.) : it automates the building process of your embedded
system and eases the cross-compilation process.

#### Why BuildRoot？===

首先是其对BeagleBone 的原生支持，不需要另行移植；其次是其大小，相较于Angstrom 与 Ubuntu 的体积来说，BuildRoot非常袖珍，除去大家都一样的bootloader 与 Kernel 不算，根文件系统只有4MB 左右，还包括了uClibc。当然我们可以在编译的时候指定第三方的编译环境，比如Ti SDK 或是 cross-N，libc 也可以使用诸如eglibc 和 标准的glibc，这都不是问题。更详细的介绍参见其官方网站 [http://buildroot.uclibc.org/](http://buildroot.uclibc.org/) ，此项目到目前为止还在维护中。

#### System requirements
Install required packages following 'Mandatory packagesin' section in document。

`$ sudo apt-get install binutils build-essential g++ gcc perl  cpio rsync ncurses-dev`

#### Config and Build
Use the beaglebone_defconfig

```bash
$ make beaglebone_defconfig
$ makemenuconfig
```
Select following options：
- Toolchain -> C library (glibc)
- System configuration -> remount root filesystem read-write during boot 此选项用于在boot后重新挂载可读写文件系统
- Target packages -> Filesystem and flash utilities 添加nfs组件从host机器上传输文件

You can also using "make linux-menuconfig | glibc-menuconfig | busybox-menuconfig" to config linux kernel or glibc or busybox.Then you can begin download source by `$ make source`

> Tips:由于国内网络在下载am33x-cm3-AM335xPSP时会比较慢，可以自行下载am33x-cm3-AM335xPSP_04.06.00.10-rc1.tar.gz对应文件到/buildroot/buildroot-2013.11/dl目录。重新make source就会使用次工具包。同理在下载ti linux时，为了节省时间也可以自行下载源文件打包，放置到此目录。

The make command will generally perform the following steps:

* download source files (as required);
* configure, build and install the cross-compiling toolchain using the appropriate toolchain backend, or simply import an external toolchain;
* build/install selected target packages;
* build a kernel image, if selected;
* build a bootloader image, if selected;
* create a root filesystem in selected formats. 

#### 目录结构

Buildroot output is stored in a single directory, output/. This directory contains several subdirectories:

* **images/** where all the images (kernel image, bootloader and root filesystem images) are stored.
* **build/** where all the components except for the cross-compilation toolchain are built (this includes tools needed to run Buildroot on the host and packages compiled for the target). The build/ directory contains one subdirectory for each of these components.
* **taging/** which contains a hierarchy similar to a root filesystem hierarchy. This directory contains the installation of the cross-compilation toolchain and all the userspace packages selected for the target. However, this directory is not intended to be the root filesystem for the target: it contains a lot of development files, unstripped binaries and libraries that make it far too big for an embedded system. These development files are used to compile libraries and applications for the target that depend on other libraries.
* **target/** which contains almost the complete root filesystem for the target: everything needed is present except the device files in /dev/ (Buildroot can’t create them because Buildroot doesn’t run as root and doesn’t want to run as root). Also, it doesn’t have the correct permissions (e.g. setuid for the busybox binary). Therefore, this directory should not be used on your target. Instead, you should use one of the images built in the images/ directory. If you need an extracted image of the root filesystem for booting over NFS, then use the tarball image generated in images/ and extract it as root. Compared to staging/, target/ contains only the files and libraries needed to run the selected target applications: the development files (headers, etc.) are not present, the binaries are stripped.
* **host/** contains the installation of tools compiled for the host that are needed for the proper execution of Buildroot, including the cross-compilation toolchain.
* **toolchain/** contains the build directories for the various components of the cross-compilation toolchain. 
    
### uboot comiple with Ti SDK
uboot comiple with Ti SDK

<pre>
make CROSS_COMPILE=arm-linux-gnueabihf- ARCH=arm distclean
xiaoyang@xiaoyang-desktop:/opt/ti-sdk-am335x/linux-devkit$ source environment-setup
[linux-devkit]:/opt/ti-sdk-am335x/board-support/u-boot-2013.01.01-psp06.00.00.00> make CROSS_COMPILE=arm-linux-gnueabihf- ARCH=arm distclean
[linux-devkit]:/opt/ti-sdk-am335x/board-support/u-boot-2013.01.01-psp06.00.00.00> ls board/ti/
am335x  am3517crane  beagle  dra7xx  evm  omap1510inn  omap2420h4  omap5912osk  omap5_evm  omap730p2  panda  sdp3430  sdp4430  tnetv107xevm
[linux-devkit]:/opt/ti-sdk-am335x/board-support/u-boot-2013.01.01-psp06.00.00.00> make O=am335x CROSS_COMPILE=arm-linux-gnueabihf- ARCH=arm am335x_evm
</pre>

编译最后会出现问题mkimage的问题，应该是ubuntu环境下TI提供的mkimage工具不兼容导致，解决办法：`$ apt-get install uboot-mkimage`

然后修改makefile将mkimage前的路径去掉，使用系统默认的mkimage，或者直接mkimage

```
$: mkimage -T omapimage -a 0x402F0400 -d /opt/ti-sdk-am335x/board-support/u-boot-2013.01.01-psp06.00.00.00/am335x/spl/u-boot-spl.bin /opt/ti-sdk-am335x/board-support/u-boot-2013.01.01-psp06.00.00.00/am335x/MLO
make O=am335x CROSS_COMPILE=arm-linux-gnueabihf- ARCH=arm am335x_evm
```

### Make the bootable MicroSD Card for Beaglebone
First Check your disk and find SD card, **/dev/sdb** for example:

```bash
$ sudo emerge dosfstools
$ sudo fdisk -l
$ export DISK=/dev/sdb
```

Then unmount the disk, and flash /dev/sdb

```bash
$ export DISK=/dev/sdc
$ sudo dd if=/dev/zero of=${DISK} bs=1M count=16
$ sudo sfdisk --in-order --Linux --unit M ${DISK} <<-__EOF__
1,48,0xE,*
,,,-
__EOF__
$ sudo mkfs.vfat -F 16 ${DISK}1 -n boot
$ sudo mkfs.ext4 ${DISK}2 -L rootfs
$ sudo mkdir -p /media/boot/
$ sudo mkdir -p /media/rootfs/
$ sudo mount ${DISK}1 /media/boot/
$ sudo mount ${DISK}2 /media/rootfs/
$ sudo cp -v MLO  /media/boot/       
$ sudo cp -v u-boot.img  /media/boot/
$ sudo cp -v ./uEnv.txt /media/boot/
$ sudo cp -v uImage  /media/boot/
$ sudo mkdir -p /media/boot/dtbs/          
$ sudo cp -v am335x-bone.dtb  /media/boot/dtbs/
$ sudo dd if=rootfs.ext4 of=${DISK}2 bs=128k
$ sync
$ sudo umount /media/boot
$ sudo umount /media/rootfs
$ exit
```

Following pages may helpful:
[http://linux-sunxi.org/Bootable_SD_card](http://linux-sunxi.org/Bootable_SD_card)

现在SD卡被分成两个区，一个LBA vfat区放置boot相关文件，一个ext4放置文件系统等。现在将文件系统强制拷贝到第二个分区，因为文件系统带有分区信息，因此可以直接dd拷贝1


```$ sudo dd if=rootfs.ext2 of=/dev/sdc2 bs=128k```

因为uImage放到了第一个分区中才导致uBoot加载不了kernel，但可以使用uEnv.txt进行启动的配置，uEnv.txt拷贝到第一分区，内容如下：
```uenvcmd=if fatload mmc 0 0x80200000 uImage; then run mmcboot; fi;```

### USB-Serial Driver for Beaglebone on ubuntu12.04LTS
为了使用beaglebone上的usb串口，需要简单的写一个驱动挂载规则，下面的文件是从angstrome中拷贝过来的，sudo执行后usb连接beaglebone开发板即可看到/dev/ttyUSB0串口。

```bash
cat > /etc/udev/rules.d/73-beaglebone.rules <<EOF
ACTION=="add", SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_interface", \
        ATTRS{idVendor}=="0403", ATTRS{idProduct}=="a6d0", \
        DRIVER=="", RUN+="/sbin/modprobe -b ftdi_sio"

ACTION=="add", SUBSYSTEM=="drivers", \
        ENV{DEVPATH}=="/bus/usb-serial/drivers/ftdi_sio", \
        ATTR{new_id}="0403 a6d0"

ACTION=="add", KERNEL=="ttyUSB*", \
    ATTRS{interface}=="BeagleBone", \
        ATTRS{bInterfaceNumber}=="00", \
    SYMLINK+="beaglebone-jtag"

ACTION=="add", KERNEL=="ttyUSB*", \
    ATTRS{interface}=="BeagleBone", \
        ATTRS{bInterfaceNumber}=="01", \
    SYMLINK+="beaglebone-serial"
EOF

sudo udevadm control --reload-rules
```

### Using Sitara Linux SDK
 - [Sitara\_Linux\_Software\_Developer\_Guide](http://processors.wiki.ti.com/index.php/Sitara_Linux_Software_Developer%E2%80%99s_Guide)
 - [BuildRoot FOR BeagleBone](http://my.oschina.net/u/614480/blog/105059)
 - [Linux on ARM BeagleBone](http://eewiki.net/display/linuxonarm/BeagleBone) 
