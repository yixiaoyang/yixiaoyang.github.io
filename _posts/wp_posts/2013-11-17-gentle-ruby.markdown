---
author: leon
comments: true
date: 2013-11-17 16:08:15+00:00
layout: post
slug: '%e4%bc%98%e9%9b%85-ruby'
title: 优雅? Ruby!
wordpress_id: 186
categories:
- Ruby
tags:
- Ruby
- 社区
- 程序猿
---

在社区里面提到一个stackoverflows上的问题：写一个triangle(a,b,c)函数，判断是等边、等腰还是一般三角形，并且能在有一边小于等于零或两边之和小于第三边的情况抛出error。

C风格（普通程序员）

    
    def triangle(a, b, c)
      # WRITE THIS CODE
      if a==b && a==c
        return :equilateral
      end
      if (a==b && a!=c) || (a==c && a!=b) || (b==c && b!=a)
        return :isosceles
      end
      if a!=b && a!=c && b!=c
        return :scalene
      end
      if a==0 && b==0 && c==0
        raise new.TriangleError
      end
    end
    
    # Error class used in part 2.  No need to change this code.
    class TriangleError < StandardError
    end


支持人数最高的：

    
    def triangle(a, b, c)
      raise TriangleError if [a,b,c].min <= 0
      x, y, z = [a,b,c].sort
      raise TriangleError if x + y <= z
      [:equilateral,:isosceles,:scalene].fetch([a,b,c].uniq.size - 1)
    end


卧槽，你不要欺负我，才学了没两个月的愣头青最后一句愣是没看懂。。。

    
     [:equilateral,:isosceles,:scalene].fetch([a,b,c].uniq.size - 1)


于是为了照顾咱们这些新手，社区有人提出了如下版本：

    
    triangle(X,Y,Z) ->
      case lists:usort([X,Y,Z]) of
        [0|_]                 -> error;
        [_A,_B]               -> isosceles;
        [_A]                  -> equilateral;
        [A,B,C] when C >= A+B -> error;
        _                     -> scalene
      end.


当然，这个问题不涉及算法复杂度，只是侧面地提醒一下编程习惯，以及，Ruby是多么简洁优雅的现代语言，c转过来的伤不起。

原问题：[http://stackoverflow.com/questions/3834203/ruby-koan-151-raising-exceptions](http://stackoverflow.com/questions/3834203/ruby-koan-151-raising-exceptions)

社区贴：[http://ruby-china.org/topics/15476?page=1#replies](http://ruby-china.org/topics/15476?page=1#replies)
