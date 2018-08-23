---
author: leon
comments: true
date: 2015-8-2 19:30:25+00:00
layout: post
title: '[CPP] Boost库概要' 
categories:
- c++
tags:
- c++
---


Boost和STL库是C++的血肉，C++的语言特性本身则是筋骨。现在在用的Qt库本身大而全，用起来方便，最重要的是跨平台，嵌入式用户交互方面的不二选择。

目前使用Boost库的需求不是很充分，因为Qt在大部分应用上都可以胜任。不过Boost以效率和稳定颇有声誉，作为C++标准化新特性的重要参考，可略作了解。

### 文本处理

#### Boost.Regex

解决大量匹配问题的正则表达式库。不必羡慕perl、awk和sed。

#### Boost.Spirit & Boost.Tokenizer

Spirit是一个语法生成框架，可以使用其构建命令分析器以及简单的语言处理器。想想Bison和flex就明白了。

#### Boost.String_algo

可以视为String类的扩展算法工具集。提供大小写转换、空格清除、分割、查找替换等扩展算法工具。

###容器和算法

一些比较经典常用的数据结构和算法的实现。

#### Boost.Array

有了Vector这玩意儿然并卵。除了基本的数组功能，提供类型安全而不牺牲性能，=_=，窃以为C++很多安全性的封装如智能指针，让用它的愚蠢程序员更加愚蠢，让用它的聪明程序员变得愚蠢。

#### Boost.MultiArray

多维容器。做矩阵计算的一定会喜欢。Vector二维以上数组声明都觉得麻烦。

#### Boost.Graph

泛型图算法工具集，包含邻接链表、邻接矩阵和边列表。提供Dijsktra最短路径算法，K最小生成树算法、拓扑逻辑排序等经典算法。


#### Boost.Multi-index

为底层容器提供多个索引，一个底层的容器可以有不同的排序方法和不同的访问语义。当std::set 和 std::map 不够用时,就可以用Boost.Multi-index,通常是在需要为查找元素而维护多个索引时。
不太了解，做稍微复杂的数据索引时应该会用到。


#### Boost.Range

用过Python/Ruby的Rang后秒懂。

#### Boost.Tunple

STL的pair不支持n-tunples，一般用自定义struct或者class实现，这个类模板支持直接声明和使用,如函数返回类型或参数,并提供一个泛型的方法来访问tuple的元素。

#### Boost.Variant

泛型类型的操作和转换。Qt中有类似的Qvariant。

###函数对象

#### Boost.bind

**bind简介**

这个是重头戏，bind、函数对象、signal/slot等特性经常会用。
我觉得bind最重要的作用是：`改变参数的顺序`，这使得我们可以少创建一些类似功能参数不同或者参数冗余的函数对象，减少代码量，使行为局部化。虽然标准库已经提供了一些可用的工具,如 bind1st 和 bind2nd, 但是这不够用，bind由此产生。

Bind是对标准库的绑定器 bind1st和bind2nd的泛化,返回值是一个函数对象。

**bind范例**

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "boost/bind.hpp"

class status {
    std::string name;
    bool ok;
public:
    status(const std::string& name):name(name),ok(true) {}
    void break_it() {
        ok=false;
    }
    bool is_broken() const {
        return ok;
    }
    void report() const {
        std::cout << name << " is " <<
                    (ok ? "working nominally":"terribly broken") << '\n';
    }
};

void bind_test()
{
    std::vector<status> statuses;
    statuses.push_back(status("status 1"));
    statuses.push_back(status("status 2"));
    statuses.push_back(status("status 3"));
    statuses.push_back(status("status 4"));
    statuses[1].break_it();
    statuses[2].break_it();

#if 0
    // 这个循环正确地完成了任务,但它是冗长、低效的(由于要多次调用statuses.end()),并且不象使用标准库算
    // 法for_each那样清楚地表明意图。为了用for_each 来替换这个循环,我们需要用一个适配器来对vector元素
    // 调用成员函数report。这时,由于元素是以值的方式保存的,我们需要的是适配器mem_fun_ref
    for (std::vector<status>::iterator it=statuses.begin();
        it!=statuses.end();++it) {
        it->report();
    }
#else
    // for_each(str.begin(), str.end(), (mem_fun(&funtest::doSth),this));
    // 把str中的每个元素作为参数送到第三个参数（是个函数的指针）所指向的函数。
    std::for_each( statuses.begin(),statuses.end(),
                    boost::bind(&status::report,_1));
#endif
}

int main(int , char **)
{
    bind_test();
    return 0;
}
```

上面`std::for_each( statuses.begin(),statuses.end(),boost::bind(&status::report,_1));`中的那一个占位符让人很迷惑。
实际上，类成员函数的绑定不同于匿名函数的绑定，成员函数必须有一个父对象或者指针，才能经过this进行调用，而匿名函数可以直接进行()操作,
因此bind需要“牺牲”一位占位符要求用户提供类的实例、指针或者引用，用过对象作为第一个参数来调用成员函数。
即`bind(&x::func,x,_1,_2...)`最多只能绑定8位参数。

假设我们有一个类

```cpp
class ClassA{
public:
    int f(int a, int b){return a+b;}
};
```
以下三种绑定都是成立的

```
ClassA obj, &obj_ref=obj;
ClassA *obj_ptr = &obj

cout << bind(&obj::f, obj, _1, 20)(10) << endl;
cout << bind(&obj::f, obj_ref, _1, _2)(10,20) << endl;
cout << bind(&obj::f, obj_ptr, _1, _2)(10,20) << endl;

```

**bind的内部实现**

bind的实现主要依靠：

   1. 对()运算符的重载
   2. 空间内置占位符变量
   3. 函数对象的调用
   4. 泛型支持

看了实现解析并没有说的那么`黑魔法`。要说黑魔法，lamba才算是。

**其他**

1. boost::bind() 返回的函数对象会保存要绑定的实参. 而且总是拷贝一份以值的方式保存..


#### Boost.signals2

signals2基于signals实现的一个线程安全信号/插槽模型。主要功能类似Qt的信号/插槽模型，对于事件处理系统相当有用。

### 并发处理

#### thread

支持最广泛的windows和posix线程，所以跨平台能力比较好，包含创建销毁等待、线程（组）管理、同步等基本功能，多线程封装的库虽然能够简化编程过程，但是写好并发程序其实是另外一回事。就像一个拿好刀的厨师不一定能做出一手好菜一样。

#### asio

asio主要关注网络通信。使用大量的类封装了socket api，提供了一些cpp接口的网络编程接口，支持tcp/icmp/udp等常用协议。总而言之，使用asio可以大量简化、可靠的socket同步/异步编程类，比用C从最小服务器/客户端程序写起要好很多，我们不需要一遍遍造毫无创意的轮子。

### 其他

有个支持python的扩展库--写过python的人应该直到python可以用cpp扩展，两者的区别就一个是在cpp程序里调用python程序，一个是使用cpp实现/重写自定义python组件并调用。

### 关于boost和Cpp

Boost库给总是给我一种古怪的感觉。cpp是一门糟糕的设计语言，但是在很多系统层应用场合，无法使用更加高级的解释性语言的情况下，cpp比c更有生产力而不输效率。

参考书籍:

- Boost documents
- Boost程序库完全开发指南
- Beyond the C++ Standard Library

