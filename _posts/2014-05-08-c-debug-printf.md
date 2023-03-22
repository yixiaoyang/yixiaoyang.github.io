---
author: leon
comments: true
date: 2014-05-08 10:00:05+00:00
layout: post
math: true
title: '[C] Debug Printf和大小端测试范例' 
categories:
- c
tags:
- c
---


#### Endien Test and Debug

{% highlight c linenos %}
    #include<stdio.h>

    #define DEBUG_LEVEL_FATEL   (0)
    #define DEBUG_LEVEL_ERROR   (1)
    #define DEBUG_LEVEL_WARN    (2)
    #define DEBUG_LEVEL_INFO    (3)
    #define DEBUG_LEVEL_DEBUG   (4)
    #define DEBUG_LEVEL_FLOW    (5)
    #define DEBUG_LEVEL_MAX     (6)

    #define debug_print(level, fmt, args...) \
        do {    \
            if (_debug_level > level ){ \
                printf("[%d %s %16s] %05d: " fmt "\n", level, __FILE__, __FUNCTION__ , __LINE__, ##args);\
            }\
        }while (0)

    static _debug_level = DEBUG_LEVEL_DEBUG;


    int main()
    {
    unsigned int uiTest;
    unsigned char *pucTmp = NULL;

    uiTest = 0x12345678;
    pucTmp = (unsigned char *)&uiTest; 

    if(*pucTmp == 0x78)
    {
        printf("This is Little Endian\n");
    }
    else
    {
        printf("This is Big Endian\n");
    }

    debug_print(DEBUG_LEVEL_DEBUG, "hello1");
    debug_print(DEBUG_LEVEL_FATEL, "hello2");
    debug_print(DEBUG_LEVEL_INFO, "hello3");
    debug_print(DEBUG_LEVEL_WARN, "hello4:%d", 123456);
    return 0;
    }
{% endhighlight %}

#### Debug Printf

{% highlight c linenos %}
    /*
    * Output a debug text when condition "cond" is met. The "cond" should be
    * computed by a preprocessor in the best case, allowing for the best
    * optimization.
    */
    #define debug_cond(cond, fmt, args...)      \
        do {                    \
            if (cond)           \
                printf(fmt, ##args);    \
        } while (0)

    #define debug(fmt, args...)         \
        debug_cond(_DEBUG, fmt, ##args)

    /*
    * An assertion is run-time check done in debug mode only. If DEBUG is not
    * defined then it is skipped. If DEBUG is defined and the assertion fails,
    * then it calls panic*( which may or may not reset/halt U-Boot (see
    * CONFIG_PANIC_HANG), It is hoped that all failing assertions are found
    * before release, and after release it is hoped that they don't matter. But
    * in any case these failing assertions cannot be fixed with a reset (which
    * may just do the same assertion again).
    */
    void __assert_fail(const char *assertion, const char *file, unsigned line,
            const char *function);
    #define assert(x) \
        ({ if (!(x) && _DEBUG) \
            __assert_fail(#x, __FILE__, __LINE__, __func__); })

    #define error(fmt, args...) do {                    \
            printf("ERROR: " fmt "\nat %s:%d/%s()\n",       \
                ##args, __FILE__, __LINE__, __func__);      \
    } while (0)

{% endhighlight %}
