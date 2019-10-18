---
author: leon
comments: true
date: 2019-10-09-+ 10:09:00+00:00
layout: post
title: 'C++邪教徒的自我修养-《Modern C++ Design》笔记'
categories:
- Booklist

tags:
- books
---

# Chapter 1. Policy-Based Class Design

## 1.2 Policies and Policy Classes 
```c++
template <class T> 
struct OpNewCreator 
{ 
   static T* Create() 
   { 
      return new T; 
   } 
}; 

template <class T>
struct MallocCreator
{ 
   static T* Create()
   {
      void* buf = std::malloc(sizeof(T));
      if (!buf) return 0; 
      return new(buf) T; 
   } 
};

template <class T>
struct PrototypeCreator 
{
private:
    T* pPrototype_; 

public:
   PrototypeCreator(T* pObj = 0) :pPrototype_(pObj){
   }

   T* Create(){
       return pPrototype_ ? pPrototype_->Clone():0;
   }

   T* GetPrototype() { return pPrototype_; }
   void SetPrototype(T* pObj) { pPrototype_ = pObj;}
};
```
However, they all define a function called Create with the required return type, so 
they conform to the Creator policy. 

```c++
// Library code 
template <class CreationPolicy> 
class WidgetManager : public CreationPolicy 
{
};
```

When instantiating the WidgetManager template, the client passes the desired policy: 

```c++
// Application code 
typedef WidgetManager< OpNewCreator<Widget> > MyWidgetMgr; 
```

##  Implementing Policy Classes with Template Template Parameters 

```c++
// Library code 
template <template <class c> class CreationPolicy> 
class WidgetManager : public CreationPolicy<Widget> 
{
}; 
typedef WidgetManager<OpNewCreator> MyWidgetMgr; 
```

> 搞template template不仅仅为了方便简洁，一些情况下为了在内部访问Create这个模板不得不这么做。挺复杂，慎用。

```c++
template <template <class> class CreationPolicy> 
class WidgetManager : public CreationPolicy<Widget> 
{ 
   ... 
   void DoSomething() 
   { 
      Gadget* pW = CreationPolicy<Gadget>().Create(); 
      ... 
   } 
}; 
```


## 1.7 Destructors of Policy Classes 

```c++
typedef WidgetManager<PrototypeCreator> 
   MyWidgetManager; 
   //...
MyWidgetManager wm; 
PrototypeCreator<Widget>* pCreator = &wm; // dubious, but legal 
delete pCreator;  // compiles fine, but has undefined behavior 
```

Defining a virtual destructor for a policy, however, works against its static nature and hurts performance. Many policies don't have any data members, but rather are purely behavioral by nature. The first virtual function added incurs some size overhead for the objects of that class, so the virtual destructor should be avoided.


引入类继承时析构函数问题。 如果不需要基类对派生类及对象进行操作,则不能定义虚函数,因为这样会增加内存开销.当类里面有定义虚函数的时候,编译器会给类添加一个虚函数表,里面来存放虚函数指针,这样就会增加类的存储空间.所以,只有当一个类被用来作为基类的时候,才把析构函数写成虚函数.


A solution is to have the host class use protected or private inheritance when deriving from the policy class. However, this would disable enriched policies as well (Section 1.6). 

The lightweight, effective solution that policies should use is to define a nonvirtual protected destructor: 
```c++
template <class T> 
struct OpNewCreator 
{ 
    static T* Create() 
    { 
        return new T; 
    } 
protected: 
   ~OpNewCreator() {} 
};
```



# Chapter 2. Techniques 

In this chapter you will get acquainted with the following techniques and tools: 
- Compile-time assertions 
- Partial template specialization 
- Local classes 
- Mappings between types and values (the Int2Type and Type2Type class templates) 
- The Select class template, a tool that chooses a type at compile time based on a Boolean condition 
- Detecting convertibility and inheritance at compile time 
- TypeInfo, a handy wrapper around std::type_info 
- Traits, a collection of traits that apply to any C++ type 

## 2.1 Compile-Time Assertions 

运行时断言

```c++
template <class To, class From> 
To safe_reinterpret_cast(From from) 
{ 
   assert(sizeof(From) <= sizeof(To)); 
   return reinterpret_cast<To>(from); 
} 

// application
int i = 0; 
char* p = safe_reinterpret_cast<char*>(i); 
```

The simplest solution to compile-time assertions (Van Horn 1997), and one that works in C as well as in C++, relies on the fact that a zero-length array is illegal. 

利用0数组让编译器编译时警告或者报错，取决于编译器的行为。

```c++
#define STATIC_CHECK(expr) { char unnamed[(expr) ? 1 : 0]; } 

template <class To, class From> 
To safe_reinterpret_cast(From from) 
{ 
   STATIC_CHECK(sizeof(From) <= sizeof(To)); 
   return reinterpret_cast<To>(from); 
} 
 
void* somePointer = something; 
char c = safe_reinterpret_cast<char>(somePointer); 
```

一种更好利用template的方法

A better solution is to rely on a template with an informative name; with luck, the compiler will mention  the name of that template in the error message. 

```c++
template<bool> struct CompileTimeError; 
template<> struct CompileTimeError<true> {}; 

#define STATIC_CHECK(expr) \ 
   (CompileTimeError<(expr) != 0>()) 
```

CompileTimeError is a template taking a nontype parameter (a Boolean constant). Compile-TimeError is defined only for the true value of the Boolean constant. If you try to instantiate `CompileTimeError<false>`, the compiler utters a message such as `Undefined specialization CompileTimeError<false>.` This message is a slightly better hint that the error is intentional and not a compiler or a program bug. 


## 2.2 Partial Template Specialization 

偏模板，放弃。

## 2.3 Local Classes 

可在函数内定义local类。
```c++
void Fun() 
{ 
   class Local 
   { 
      ... member variables ... 
      ... member function definitions ... 
   }; 
   ... code using Local ... 
}
```
There are some limitations—local classes **cannot define static member variables** and **cannot access nonstatic local variables**. What makes local classes truly interesting is that you can use them in template functions. Local classes defined inside template functions can use the template parameters of the enclosing function. 

```c++
class Interface 
{ 
public: 
   virtual void Fun() = 0; 
}; 

template <class T, class P> 
Interface* MakeAdapter(const T& obj, const P& arg) 
{ 
   class Local : public Interface 
   { 
   public: 
      Local(const T& obj, const P& arg) 
         : obj_(obj), arg_(arg) {} 
      virtual void Fun() 
      { 
         obj_.Call(arg_); 
      } 
   private: 
      T obj_; 
      P arg_; 
   }; 
   return new Local(obj, arg); 
}
```

## 2.4 Int2Type

```c++
template <int v>
struct Int2Type
{
enum { value = v };
};
```
我们很容易做到的一点是哪段代码执行哪段代码不执行，那如何做编译期阶段的判断（if else）。要做到这一点就是制作能让编译器认识的值，而编译认识的值就是类型，也就是说，编译器只认识类型，对它来说类型1和类型2是不一样的。这就是模板的精髓，它是类型的抽象，而类型是对象的抽象。(真。面向编译器编程)

```c++
template <typename T, bool isPolymorphic>
class NiftyContainer
{
private:
   void DoSomething(T* pObj, Int2Type<true>)
   {
   T* pNewObj = pObj->Clone();
   ... polymorphic algorithm ...
   }

   void DoSomething(T* pObj, Int2Type<false>)
   {
   T* pNewObj = new T(*pObj);
   ... nonpolymorphic algorithm ...
   }
public:
   void DoSomething(T* pObj)
   {
      DoSomething(pObj, Int2Type<isPolymorphic>());
   }
};
```

## 2.5 Type-to-Type Mapping

（放弃）

## 2.6 Type Selection

```c++
template <bool flag, typename T, typename U>
struct Select
{
  typedef T Result;
};
template <typename T, typename U>
struct Select<false, T, U>
{
  typedef U Result;
};
```
you need to store either a `vector<T*>` (if `isPolymorphic` is true ) or a `vector<T>` (if `isPolymorphic` is false ). In essence, you need a typedef ValueType that is either T* or T, depending on the value of isPolymorphic 

```c++
template <typename T, bool isPolymorphic>
class NiftyContainer
{
typedef typename Select<isPolymorphic, T*, T>::Result ValueType;
};
```

## 2.8 A Wrapper Around type_info
Standard C++ provides the `std::type_info` class, which gives you the ability to investigate object types at runtime. You typically use `type_info` in conjunction with the typeid operator. The typeid operator returns a reference to a type_info object:

```c++
void Fun(Base* pObj)
{
   // Compare the two type_info objects corresponding
   // to the type of *pObj and Derived
   if (typeid(*pObj) == typeid(Derived))
   {
   }
}
```

## 2.10 Type Traits
模板特征，弃

# 3. Typelists
类型列表相关，弃

# Chapter 4. Small-Object Allocation
小对象allcator的实现，走的还是chunk表的路子。


# Chapter 5. Generalized Functors

In brief, a generalized functor is any processing invocation that C++ allows, encapsulated as a typesafe first-class object. In a more detailed definition, a generalized functor
- Encapsulates any processing invocation because it accepts pointers to simple functions, pointers to member functions, functors, and even other generalized functors—together with some or all of their respective arguments.
- Is typesafe because it never matches the wrong argument types to the wrong functions
- Is an object with value semantics because it fully supports copying, assignment, and pass by value. 
- A generalized functor can be copied freely and does not expose virtual member functions

## 5.1 The Command Design Pattern
```c++
Command resizeCmd(
   window,           // Object
   &Window::Resize,  // Member function
   0, 0, 200, 100    // Arguments
);

// Later on... Resize the window
resizeCmd.Execute(); 
```

# 参考
- The Small Object Allocation Optimization and Implement，by Encapsulating in C++