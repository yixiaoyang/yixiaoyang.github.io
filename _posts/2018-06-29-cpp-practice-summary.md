---
author: leon
comments: true
date: 2018-05-29 16:28:00+00:00
layout: post
title: '[c++]C++工程常用高级特性小本子'
categories:
- c++
tags:
- c++
---

<!-- TOC -->

- [参考列表](#)
- [内存](#)
    - [new/delete/malloc/free](#new-delete-malloc-free)
    - [智能指针](#)
        - [unique_ptr](#unique-ptr)
        - [shared_ptr](#shared-ptr)
        - [waek_ptr](#waek-ptr)
- [函数](#)
    - [构造函数和析构函数](#)
    - [仿函数](#)
    - [lambda](#lambda)
    - [函数式编程惯用语](#)
    - [虚函数](#)
    - [bind/function](#bind-function)
- [面向对象程序设计模型](#)
- [一些工程性问题](#)
    - [二进制兼容性](#)
        - [兼容性问题是怎么来的](#)
        - [哪些做法多半是安全的](#)

<!-- /TOC -->

# 参考列表
- Discovering Modern CPP
- Effective Modern C++
- Effective C++
- C++标准/Working Draft, Standard for Programming Language C++
- C++工程实践经验谈 by陈硕(giantchen@gmail.com)


# 内存


## new/delete/malloc/free

在c++中，尽量使用new/delete代替malloc/free，两者区别主要体现在构造/析构函数调用行为、对申请失败的处理上。

```c++
Header <new> synopsis
namespace std {
class bad_alloc;
class bad_array_new_length;
struct nothrow_t {};
extern const nothrow_t nothrow;
typedef void (*new_handler)();
new_handler get_new_handler() noexcept;
new_handler set_new_handler(new_handler new_p) noexcept;
}

void* operator new(std::size_t size);
void* operator new(std::size_t size, const std::nothrow_t&) noexcept;
void operator delete(void* ptr) noexcept;
void operator delete(void* ptr, const std::nothrow_t&) noexcept;
void* operator new[](std::size_t size);
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept;
void operator delete[](void* ptr) noexcept;
void operator delete[](void* ptr, const std::nothrow_t&) noexcept;
void* operator new (std::size_t size, void* ptr) noexcept;
void* operator new[](std::size_t size, void* ptr) noexcept;
void operator delete (void* ptr, void*) noexcept;
void operator delete[](void* ptr, void*) noexcept;
```

> Except for a form called the placement new, the new operator denotes a request for memory allocation on a process's [heap](https://en.wikipedia.org/wiki/Heap_(programming)). If sufficient memory is available, new initialises the memory, calling object constructors if necessary, and returns the address to the newly allocated and initialised memory.
>
> If not enough memory is available in the free store for an object of type T, the new request indicates failure by throwing an [exception](https://en.wikipedia.org/wiki/Exception_handling "Exception handling") of type std::bad_alloc. This removes the need to explicitly check the result of an allocation.


对于C++来说new分三步步：  
1. 申请一块内存(operator new), 相当于malloc  
2. 调用构造函数(placement new)  
3. 内存分配失败抛出bad_alloc异常。  

而delete的操作分两步  
1. 调用析构函数  
2. 释放内存，相当于free  

## 智能指针

> Three new smart-pointer types are introduced in C++11: `unique_ptr`, `shared_ptr`, and `weak_ptr`.
The already existing smart pointer from C ++03 named auto_ptr is generally considered as a
failed attempt on the way to unique_ptr since the language was not ready at the time. It
should not be used anymore. All smart pointers are defined in the header <memory>. 

> 《Effective modern C++》
> Item 21: Prefer std::make_unique and std::make_shared to direct use of new.

智能指针的主要实现技术是引用计数，当引用指针的对象数为0时才释放内存，从而避免double free风险。

### unique_ptr

no copy allowed, can only be moved.

```c++
unique_ptr < double > dp2 { dp } ; // Error : no copy allowed
dp2 = dp ;  // ditto

unique_ptr < double > dp2 { move ( dp )} , dp3 ;
dp3 = move ( dp2 ) ;
```

### shared_ptr
```c++
shared_ptr < double > f ()
{
  shared_ptr < double > p1 { new double };
  shared_ptr < double > p2 { new double }, p3 = p2 ;
  cout  << " p3 . use_count () = " << p3 . use_count () << endl ;
  return p3 ;
}

int main ()
{
  shared_ptr < double > p= f ();
  cout << "p. use_count () = " << p. use_count () << endl ;
}
```

If possible, a `shared_ptr` should be created with `make_shared`:
```
shared_ptr < double > p1 = make_shared < double >() ;
```
Then the management and business data are stored together in memory —and the memory caching is more efficient.

### waek_ptr

避免weak_ptr的出现，人为排除shared pointer的循环引用。

（为了给循环引用的shared_ptr擦屁股用，懒得细究）

# 函数

## 构造函数和析构函数
着重关注一下构造和析构函数。

> C ++ has six methods (four in C ++03) with a default behavior:
> - Default constructor  
> - Copy constructor  
> - Move constructor (C ++11 or higher)  
> - Copy assignment  
> - Move assignment (C++11 or higher)  
> - Destructor

**explicit构造函数**： 防止构造函数的隐式转换带来的错误或者误解。

**move构造函数**：语法特性来提高效率减少拷贝动作。（除非不得已，还是在算法上提升效率吧，move语义的效果也很鸡肋，除非有大块内存的拷贝可以考虑使用，另外，参考swap语义）。

```c++
class vector
{
    vector & operator =( vector && src )
    {
        assert ( my_size == 0 || my_size == src.my_size);
        std :: swap ( data , src . data );
        return * this ;
    }
};
```

构造和析构函数使得c++类有一个重要特点：`Resource Acquisition Is Initialization (RAII)` ，对象在生命周期内完成资源的自我管理（恩，对自己负责原则，Single Responsibility Principle）。多个对象引用同一块内存的情况可配合`shared_ptr`效果更佳。


## 仿函数
仿函数(functor)：就是类中实现一个operator()，这个类就有了类似函数的行为。  
使用场景：把通用的功能块写成全局函数或者定位为类成员都不合适的情况下使用。

## lambda
匿名函数语法糖，编译器会给你写一个仿函数出来，权当为了使std::algorithm更可用的一点补充。

> C++本身的问题是：没有lambda的话，函数对象的定义太麻烦了！你得定义一个类，重载operator()，然后再创建这个类的实例…所以lambda表达式可以看成是函数对象的语法糖，在你需要的时候，它可以很简洁地给你生成一个函数对象。

## 函数式编程惯用语
- Filter :remove unwanted items
- map :apply a transformation
- Reduce :reduce sequences to a single value

## 虚函数

先写一个demo看虚函数的行为。

```c++
#include <iostream>
#include <iomanip>

#define PRINT_FUCCTION_MESSAGE() std::cout << std::setw(10) << typeid(this).name() \
                                           << std::setw(10) <<  "@" << __FUNCTION__  \
                                           << std::endl

class Base{
  public:
    Base(){PRINT_FUCCTION_MESSAGE();}
    void func(){PRINT_FUCCTION_MESSAGE();}
    virtual void vfunc(){PRINT_FUCCTION_MESSAGE();}
    virtual ~Base(){PRINT_FUCCTION_MESSAGE();}
};

class A: public Base{
  public:
    A(){PRINT_FUCCTION_MESSAGE();}
    void func(){PRINT_FUCCTION_MESSAGE();}
    virtual void vfunc(){PRINT_FUCCTION_MESSAGE();}
    ~A(){PRINT_FUCCTION_MESSAGE();}
};

class B: public Base{
  public:
    B(){PRINT_FUCCTION_MESSAGE();}
    void func(){PRINT_FUCCTION_MESSAGE();}
    virtual void vfunc(){PRINT_FUCCTION_MESSAGE();}
    ~B(){PRINT_FUCCTION_MESSAGE();}
};

int main (int, char**){
    std::cout << "----------------------------" << std::endl;
    Base* base = new Base;
    A* a = new A;
    B* b = new B;

    std::cout << "----------------------------" << std::endl;
    base->func();
    a->func();
    b->func();

    std::cout << "----------------------------" << std::endl;
    base->vfunc();
    a->vfunc();
    b->vfunc();

    std::cout << "----------------------------" << std::endl;
    Base* bb = static_cast<Base*>(a);
    bb->vfunc();
    bb->func();

    std::cout << "----------------------------" << std::endl;

    delete base;
    delete a;
    delete b;
    return 0;
}
```

输出：
```
----------------------------
    P4Base         @Base
    P4Base         @Base
       P1A         @A
    P4Base         @Base
       P1B         @B
----------------------------
    P4Base         @func
       P1A         @func
       P1B         @func
----------------------------
    P4Base         @vfunc
       P1A         @vfunc
       P1B         @vfunc
----------------------------
       P1A         @vfunc
    P4Base         @func
----------------------------
    P4Base         @~Base
       P1A         @~A
    P4Base         @~Base
       P1B         @~B
    P4Base         @~Base
按 <RETURN> 来关闭窗口...
```

在写接口时虚函数比较有用，可以规定子类需要重载的函数，实现多态。
另外，此处的继承需要注意构造函数和析构函数的调用顺序，防止出现资源double free的情况。

> 为什么 C++ 中使用虚函数时会影响效率？
>
> 调用虚函数的时候，首先根据对象里存储的虚函数表指针vptr，找到虚函数表vtables，再根据偏移量找到哪一项，再找到虚函数地址

不知道什么时开始虚函数的实现被面试官频繁地问到，真的有用么。
> <<Master Mordern C++>> 6.1.3 Virtual Functions and Polymorphic Classes
> 
> To realize these dynamic function calls, the compiler maintains Virtual Function Tables
> (a.k.a. Virtual Method Tables) or Vtables. They contain function pointers through which
> each virtual method of the actual object is called. The reference pr has the type person&
> and refers to an object of type student. Through the vtable of pr, the call of all_info() is
> directed to student::all_info. This indirection over function pointers adds some extra cost
> to virtual functions which is significant for tiny functions and negligible for sufficiently large
> functions.


按照the C++PL的说法，虚函数调用比普通成员函数慢至多25%

## bind/function
> Scott Meyers 的 Effective C++ 3rd 第 35 条款提到了以 boost::function和 boost:bind 取代虚函数的做法
boost::function可以指向任何函数,包括成员函数。当用bind把某个成员函数绑到某个对象上时,我们得到了一个 closure(闭包)。

```c++
#include <iostream>
#include <iomanip>
#include <functional>

#define PRINT_FUCCTION_MESSAGE() std::cout << std::setw(10) << typeid(this).name() \
                                           << std::setw(10) <<  "@" << __FUNCTION__  \
                                           << std::endl
class button
{
public:
    std::function<void()> onClick;
};

class player
{
public:
    void play(){ PRINT_FUCCTION_MESSAGE();}
    void stop(){ PRINT_FUCCTION_MESSAGE();}
};

button playButton, stopButton;
player thePlayer;

void connect()
{
    playButton.onClick = std::bind(&player::play, &thePlayer);
    stopButton.onClick = std::bind(&player::stop, &thePlayer);
}

int main(int argc, char *argv[])
{
    connect();
    playButton.onClick();
    stopButton.onClick();
    return 0;
}
```
输出：
```
  P6player         @play
  P6player         @stop
```

# 面向对象程序设计模型

面向对象三板斧：封装、继承和多态。抛开面向对象不谈，工程设计上`封装`绝对在最重要的位置，良好的封装是清晰的代码层次的必要条件。继承和多态相对来说属于锦上添花的特性，能用好最好，用的不好会带来不必要的耦合，增加抽象的复杂性。

# 一些工程性问题

## 二进制兼容性

### 兼容性问题是怎么来的

这个问题在《C++工程实践经验谈 by陈硕(giantchen@gmail.com)》中被着重提起。

>有哪些情况会破坏库的 ABI到底如何判断一个改动是不是二进制兼容呢?这跟 C++ 的实现方式直接相关,虽然 C++ 标准没有规定 C++ 的 ABI,但是几乎所有主流平台都有明文或事实上的 ABI 标准。比方说 ARM 有 EABI,Intel Itanium 有 Itanium ABI 10 ,x86-64 有仿Itanium 的 ABI,SPARC 和 MIPS 也都有明文规定的 ABI,等等。x86 是个例外,它只有事实上的 ABI,比如 Windows 就是 Visual C++,Linux 是 G++(G++ 的 ABI还有多个版本,目前最新的是 G++ 3.4 的版本) ,Intel 的 C++ 编译器也得按照 VisualC++ 或 G++ 的 ABI 来生成代码,否则就不能与系统其它部件兼容。
>
> C++ ABI 的主要内容:
> - 函数参数传递的方式,比如 x86-64 用寄存器来传函数的前 4 个整数参数
> - 虚函数的调用方式,通常是 vptr/vtbl 然后用 vtbl[offset] 来调用
> - struct 和 class 的内存布局,通过偏移量来访问数据成员
> - name mangling
> - RTTI 和异常处理的实现(以下本文不考虑异常处理)
>
> C/C++ 通过头文件暴露出动态库的使用方法,这个“使用方法”主要是给编译器看的,编译器会据此生成二进制代码,然后在运行的时候通过装载器 (loader) 把可执行文件和动态库绑到一起。如何判断一个改动是不是二进制兼容,主要就是看头文件暴露的这份“使用说明”能否与新版本的动态库的实际使用方法兼容。因为新的库必然有新的头文件,但是现有的二进制可执行文件还是按旧的头文件来调用动态库。
>
>这里举一些源代码兼容但是二进制代码不兼容例子
> - 给函数增加默认参数,现有的可执行文件无法传这个额外的参数。
> - 增加虚函数,会造成 vtbl 里的排列变化。(不要考虑“只在末尾增加”这种取巧行为,因为你的 class 可能已被继承。)
> - 增加默认模板类型参数,比方说 Foo<T> 改为 Foo<T, Alloc=alloc<T> >,这会改变 name mangling
> - 改变 enum 的值,把 enum Color { Red = 3 }; 改为 Red = 4。这会造成错位。当然,由于 enum 自动排列取值,添加 enum 项也是不安全的,在末尾添加除外。
> 
> 给 class Bar 增加数据成员,造成 sizeof(Bar) 变大,以及内部数据成员的offset 变化,这是不是安全的?通常不是安全的,但也有例外。
> - 如果客户代码里有 new Bar,那么肯定不安全,因为 new 的字节数不够装下新Bar 对象。相反,如果 library 通过 factory 返回 Bar* (并通过 factory 来销毁对象)或者直接返回 shared_ptr<Bar>,客户端不需要用到 sizeof(Bar),那么可能是安全的。
> - 如果客户代码里有 Bar* pBar; pBar->memberA = xx;,那么肯定不安全,因为memberA 的新 Bar 的偏移可能会变。相反,如果只通过成员函数来访问对象的数据成员,客户端不需要用到 data member 的 offsets,那么可能是安全的。
> - 如果客户调用 pBar->setMemberA(xx); 而 Bar::setMemberA() 是个 inline func-tion,那么肯定不安全,因为偏移量已经被 inline 到客户的二进制代码里了。如果 setMemberA() 是 outline function,其实现位于 shared library 中,会随着Bar 的更新而更新,那么可能是安全的。
> 
> 那么只使用 header-only 的库文件是不是安全呢?不一定。如果你的程序用了boost 1.36.0,而你依赖的某个 library 在编译的时候用的是 1.33.1,那么你的程序和这个 library 就不能正常工作。因为 1.36.0 和 1.33.1 的 boost::function 的模板参数类型的个数不一样,后者一个多了 allocator。


### 哪些做法多半是安全的

> 前面我说“不能轻易修改”,暗示有些改动多半是安全的,这里有一份白名单,欢迎添加更多内容。只要库改动不影响现有的可执行文件的二进制代码的正确性,那么就是安全的,我们可以先部署新的库,让现有的二进制程序受益。
> - 增加新的 class
> - 增加 non-virtual 成员函数或 static 成员函数
> - 修改数据成员的名称,因为生产的二进制代码是按偏移量来访问的,当然,这会造成源码级的不兼容。
> - 还有很多,不一一列举了。
> 
> 欢迎补充

