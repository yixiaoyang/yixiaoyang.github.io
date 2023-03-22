---
author: leon_e@163.com
comments: false
date: 2014-10-13 07:49:12+00:00
layout: post
slug: beagleboneam3359%e8%8a%af%e7%89%87gpio%e5%a4%96%e9%83%a8%e4%b8%ad%e6%96%ad
title: '[Beaglebone] AM3359芯片GPIO外部中断'
wordpress_id: 352
categories:
- Beaglebone
tags:
- Beaglebone
- C/C++
- 嵌入式
- 编程
---

AM3359芯片的GPIO外部中断实现比较简单，可参照ti-sdk-am335x-evm-06.00.00.00官方SDK内核的`gpio_omap.c`中关于GPIO IRQ部分实现，基本上就是设置GPIO复用->设置input方向->使能中断->填写中断处理函数槽->清中断标记位等几个步骤。需要理解的异步通知fasync的使用，这是中断等信号传递到用户空间的常用方法。


####驱动方面

1. 在设备抽象的数据结构中增加一个`struct fasync_struct`的指针
2. 实现设备操作中的fasync函数，主体就是调用内核的`fasync_helper`函数。
3. 在需要向用户空间通知的地方(例如中断中)调用内核的`kill_fasync`函数发送信号。
4. 在驱动的release方法中调用前面定义的fasync函数fasync(-1, fd, 0)以删除释放通知。

#### 应用层方面
1. 利用signal或者sigaction设置SIGIO信号的处理函数
2. fcntl的`F_SETOWN`指令设置当前进程为设备文件owner
3. fcntl的`F_SETFL`指令设置FASYNC标志

为测试外部驱动，我让硬件工程师接了一个下拉电阻，但是拨上开关后出现了外部中断一直触发的现象，之后用示波器抓了GPIO端口的波形发现噪声太大，有时都能超过2v，遂使用一般按键的方法外接电容处理减少噪声，中断才一次按键一次中断响应。

驱动和应用层代码如下。

```c    
#include <linux/kernel.h>  
#include <linux/module.h>  
#include <linux/cdev.h>  
#include <linux/fs.h>  
#include <linux/device.h>  
#include <linux/syscalls.h>
#include <linux/interrupt.h> 
#include <linux/gpio.h>
#include <linux/of_gpio.h>
#include <linux/of_platform.h>
#include <linux/uaccess.h>  
#include <linux/string.h> 

#include <mach/gpio.h>
#include <mach/irqs.h>

#include <plat/gpio.h>
#include <plat/am33xx.h>

#define GPIO_TO_PIN(bank, gpio)     (32 * (bank) + (gpio))
#define IRQ_GPIO_PIN                GPIO_TO_PIN(1, 13)

#define   TRIGGERD_FLAG_IRQ_DISABLED        (1<<0)
#define   TRIGGERD_FLAG_IRQ_ENABLED         (1<<1)

/* cmd definations shouled be same with that in user space */
#define MAGIC_NUM   'k'
enum IOCTL_CMDS{
  IOCTL_CMD_IRQ_NONE = 0,
  IOCTL_CMD_IRQ_DISABLE = _IO(MAGIC_NUM,0),
  IOCTL_CMD_IRQ_ENABLE  = _IO(MAGIC_NUM,1),
};

struct fpga_key_dev  
{  
    struct cdev cdev;  
    dev_t devno;
    struct class *fpga_key_class; 
    struct fasync_struct *async_queue;

    int message_cdev_open;
    
    int irq_no; 
    unsigned int triggered_flags;
};

struct fpga_key_dev fpga_key_dev;

static void irq_enable( struct fpga_key_dev *dev, unsigned int enable)
{
  if(!dev)
    return ;
  
  switch(enable){
    case IOCTL_CMD_IRQ_ENABLE:
      printk(KERN_INFO "enable irq\n");
      enable_irq(dev->irq_no);
      dev->triggered_flags &= (~TRIGGERD_FLAG_IRQ_DISABLED);
      dev->triggered_flags |= TRIGGERD_FLAG_IRQ_ENABLED;  
      break;
    case IOCTL_CMD_IRQ_DISABLE:
      printk(KERN_INFO "disable irq\n");
      disable_irq_nosync(dev->irq_no);
      dev->triggered_flags &= (~TRIGGERD_FLAG_IRQ_ENABLED);
      dev->triggered_flags |= TRIGGERD_FLAG_IRQ_DISABLED;  
      break;
    default:
      break;
  }
  printk(KERN_INFO "irq_enable finished\n");
}

irqreturn_t irq_handler(int irqno, void *dev_id)
{
    struct fpga_key_dev *dev = &fpga_key_dev;
 
    printk(KERN_INFO "fpga int triggered\n");
    if (dev->async_queue)
        kill_fasync(&dev->async_queue, SIGIO, POLL_IN); 
    
    return IRQ_HANDLED;
}

static int fpga_key_open(struct inode *node, struct file *fd)  
{  
    struct fpga_key_dev *dev;
      
    dev = container_of(node->i_cdev, struct fpga_key_dev, cdev);
    
    if (!dev->message_cdev_open) {
        dev->message_cdev_open = 1;
        fd->private_data = dev;
    }
    else{
        return -EFAULT;
    }
    
    return 0;  
}   
  
static ssize_t fpga_key_write(struct file *fd, const char __user *buf, size_t len, loff_t *ptr)  
{  
    return 0;  
}  
static ssize_t fpga_key_read(struct file *fd, char __user *buf, size_t len, loff_t *ptr)  
{  
    struct fpga_key_dev *dev = fd->private_data;
    int size = sizeof(unsigned int);
    if(len < size){
      printk(KERN_ERR "read length too small");
      return EFAULT;
    }
    
    if(copy_to_user(buf, &dev->triggered_flags, size))  
        return -EFAULT;  
    return 0;  
}  

static int fpga_key_fasync(int fd, struct file *filp, int mode)
{
    struct fpga_key_dev     *dev = filp->private_data;
    fasync_helper(fd, filp, mode, &dev->async_queue); 
    return 0;
}

static int fpga_key_release(struct inode *node, struct file *fd)  
{  
    struct fpga_key_dev *dev = fd->private_data;

    dev->message_cdev_open = 0;
    fpga_key_fasync(-1, fd, 0);
    
    return 0;  
} 


static int fpga_key_ioctl(struct file *fd, unsigned int cmd, unsigned long args)  
{  
    struct fpga_key_dev *dev = fd->private_data;
  
    if(!dev){
      printk(KERN_ERR "dev data null");
      return -EINVAL;
    }
    switch(cmd)  
    {  
        case IOCTL_CMD_IRQ_NONE:  
          break;  
        case IOCTL_CMD_IRQ_DISABLE:  
          irq_enable(dev, IOCTL_CMD_IRQ_DISABLE);
          break;  
        case IOCTL_CMD_IRQ_ENABLE:
          irq_enable(dev, IOCTL_CMD_IRQ_ENABLE);
          break;
        default:  
          return -ENOTTY;  
          break;  
    }  
    return 0;  
}

struct file_operations meassage_operatons =  
{  
    .owner = THIS_MODULE,  
    .open = fpga_key_open,  
    .write = fpga_key_write,
    .read = fpga_key_read, 
    .fasync = fpga_key_fasync,
    .release = fpga_key_release,  
    .unlocked_ioctl = fpga_key_ioctl,
};  
  
static int __init fpga_key_init(void)  
{  
    struct fpga_key_dev * dev = &fpga_key_dev;  
    int ret = 0;
    int irq = 0;
    const char* gpio_lable = "fpga_msg_irq_key";
  
    dev->triggered_flags = 0;
    dev->irq_no = 0;
    
    alloc_chrdev_region(&dev->devno, 0, 1, "fpga_msg_irq");  
    cdev_init(&dev->cdev, &meassage_operatons);  
    cdev_add(&dev->cdev, dev->devno, 1);  

    dev->fpga_key_class = class_create(THIS_MODULE, "fpga_msg_irq_class");  
    if(IS_ERR(dev->fpga_key_class)) {  
         printk(KERN_ERR "failed in creating class./n");  
         goto fail1;   
     }  
    device_create(dev->fpga_key_class, NULL, dev->devno, NULL, "fpga_msg_irq_dev");  

    /* gpio interrupt request */
    ret = gpio_request(IRQ_GPIO_PIN, gpio_lable);
    if(ret){
        printk(KERN_ERR "gpio_request() failed !\n");
        goto fail1;
    }
    ret = gpio_direction_input(IRQ_GPIO_PIN);
    if(ret){
        printk(KERN_ERR "gpio_direction_input() failed !\n");
        goto fail2; 
    }
    
    irq = gpio_to_irq(IRQ_GPIO_PIN);
    if(irq < 0){
        printk(KERN_ERR "gpio_to_irq() failed !\n");
        ret = irq;
        goto fail2; 
    }
    dev->irq_no = irq;
    
#if 0
   ret = request_irq(dev->irq_no, 
                    irq_handler, 
                    IRQF_TRIGGER_RISING | IRQF_SHARED, 
                    gpio_lable, 
                    &dev->devno); 
#else
   ret = request_irq(dev->irq_no, 
                    irq_handler, 
                    IRQF_TRIGGER_RISING | IRQF_DISABLED, 
                    gpio_lable, 
                    &dev->devno); 
#endif
    
    if(ret){
        printk(KERN_ERR "request_irq() failed ! %d\n", ret);
        goto fail2;
    }
    
    printk(KERN_INFO "fpga_key_to_app_init\n");      
    return 0;  

fail2:  
    gpio_free(IRQ_GPIO_PIN);  
fail1:    
    device_destroy(dev->fpga_key_class, dev->devno);  
    class_destroy(dev->fpga_key_class);   
    cdev_del(&dev->cdev);  
    unregister_chrdev_region(dev->devno, 1);  

    return ret;
}  
static void __exit fpga_key_exit(void)  
{  
    struct fpga_key_dev *dev = &fpga_key_dev;  
  
    free_irq(dev->irq_no, &dev->devno); 
    gpio_free(IRQ_GPIO_PIN);
    
    device_destroy(dev->fpga_key_class, dev->devno);  
    class_destroy(dev->fpga_key_class);   
    cdev_del(&dev->cdev);  
    unregister_chrdev_region(dev->devno, 1);  
       
    printk(KERN_INFO "fpga_key_to_app_exit\n");  
}  
module_init(fpga_key_init);  
module_exit(fpga_key_exit);  
  
MODULE_LICENSE("GPL");  
MODULE_AUTHOR("hityixiaoyang@gmail.com");  
MODULE_DESCRIPTION("AM335x GPIO IQR for FPGA");  






/*
 * linux异步通知类似于中断，程序开始做的就是将想监测的文件设置为异步通知，然后程序就去做别的事情了，
 * 当这些文件上面有数据可读时候，就发送一个信号，程序接受到这个信号就去处理这个文件的数据。
 * 这样程序不再是主动去读文件了，而是以类型于中断的方式，这样程序更加灵活。
 * 
 * 要弄清linux异步通知必须要弄明白一下几件事:
 * 1.谁发信号
 * 2.发给谁
 * 3.接受到信号怎么办
 */
#include<stdio.h>  
#include<sys/types.h>  
#include<sys/stat.h>  
#include<fcntl.h>  
#include<sys/select.h>  
#include<unistd.h>  
#include<signal.h>  
#include<string.h>  
#include<linux/ioctl.h>

#define FPIO_IRQ_DEV_NAME   "/dev/fpga_msg_irq_dev"

#define MAGIC_NUM   'k'
enum IOCTL_CMDS{
  IOCTL_CMD_IRQ_NONE = 0,
  IOCTL_CMD_IRQ_DISABLE = _IO(MAGIC_NUM,0),
  IOCTL_CMD_IRQ_ENABLE  = _IO(MAGIC_NUM,1),
};

unsigned int flag = 0;  
int fd = 0;
  
void sig_handler(int sig)  
{  
  printf("[pid %d] %s recieved signal\n",getpid(),__FUNCTION__);  
}  

int main(void)  
{  
    int f_flags; 
    unsigned int number; 
    
    fd=open(FPIO_IRQ_DEV_NAME,O_RDWR);  
    if(fd < 0){  
        perror("open gpio irq device failed");  
        return -1;  
    }  

    signal(SIGIO, sig_handler); 
    fcntl(fd, F_SETOWN, getpid());
    
   /* 
    * 设置该文件的标志为FASYNC,设置文件的标志为FASYNC将导致驱动程序的fasync被调用
    * 这样文件就开始处于异步通知状态了
    */
    f_flags = fcntl(fd, F_GETFL);
    
    printf("waiting for signal \n"); 
    fcntl(fd, F_SETFL, FASYNC | f_flags); 

    while (1){
      read(fd, &number, sizeof(unsigned int));
      printf("key count=%d\n",number);
      sleep(15); 
    }
    
    close(fd);  

    printf("main exit\n"); 
    return 0;  
}  

```



