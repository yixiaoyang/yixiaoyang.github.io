---
author: leon
comments: true
date: 2018-05-26 10:27:00+00:00
layout: post
title: '[设计模式]一个老生常谈的问题：业务控制、逻辑和数据的分离'
categories:
- 设计模式
tags:
- 设计模式
---


> Algorithm = Logic + Control  by Robert Kowalski Imperial College, London
>
> An algorithm can be regarded as consisting of a logic component, which specifies the knowledge to be used in solving problems, and a control component, which determines the problem-solving strategies by means of which that knowledge is used. The logic component determines the meaning of the algorithm whereas the control component only affects its effkiency. The effkiency of an algorithm can often be improved by improving the control component without changing the logic of the algorithm. We argue that computer programs would be more often correct and more easily improved and modified if their logic and control aspects were identified and separated in the program text. Key Words and Phrases: control language, logic programming, nonprocedural language, programming methodology, program specification, relational data structures
> 任何算法都会有两个部分， 一个是 Logic 部分，这是用来解决实际问题的。另一个是 Control 部分，这是用来决定用什么策略来解决问题。Logic 部分是真正意义上的解决问题的算法，而 Control 部分只是影响解决这个问题的效率。程序运行的效率问题和程序的逻辑其实是没有关系的。我们认为，如果将 Logic 和 Control 部分有效地分开，那么代码就会变得更容易改进和维护。


大部分代码复杂度的原因：
- 业务逻辑本身的复杂度（Logic部分，从需求抽取出来的逻辑）
- 业务逻辑、控制逻辑（业务逻辑的表现和实现部分）、数据处理的高耦合

降低复杂度，分离控制、逻辑和数据的一些方法：
- 梳理业务，分层，分块、管道、过滤、代理
- 面向对象抽象
- 状态机
- 微内核
- DSL – Domain Specific Language领域内自治
- 设计模式：委托、策略、桥接、修饰、IoC/DIP、MVC
- 函数式编程：修饰、管道、拼装
