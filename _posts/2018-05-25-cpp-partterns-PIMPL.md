---
author: leon
comments: true
date: 2018-05-21 16:28:00+00:00
layout: post
title: '[设计模式]PIMPL惯用法分离内部实现和外部方法接口'
categories:
- 设计模式
tags:
- 设计模式
- c++
---

主要思想是使用`pointer to implementation`惯用法将类内部数据或者方法使用一个implementation隐藏起来。在 `Efficient Mordern C++`一书中 `Item 22: When using the Pimpl Idiom, define special member functions in the implementation file`.有详细讲解。

优点：数据隐藏，编译优化


```c++
//////////////////////////////////////////////////////
// Parser.h
//////////////////////////////////////////////////////
class Parser {
public:
    Parser(const char *params);
    ~Parser();
    void parse(const char *input);

private:
    class Impl;     // Forward declaration of the implementation class
    Impl *impl_;    // PIMPL
};

//////////////////////////////////////////////////////
// Parser.cpp
//////////////////////////////////////////////////////
// The actual implementation definition:
class Parser::Impl {
public:
    Impl(const char *params) {
        // Actual initialization
    }
    void parse(const char *input) {
        // Actual work
    }
};

// Create an implementation object in ctor
Parser::Parser(const char *params)
: impl_(new Impl(params))
{}

// Delete the implementation in dtor
Parser::~Parser() { delete impl_; }

// Forward an operation to the implementation
void Parser::parse(const char *input) {
    impl_->parse(input);
}


```