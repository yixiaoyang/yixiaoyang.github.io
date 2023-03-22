---
author: leon
comments: true
date: 2016-03-05 18:03:25+00:00
layout: post
math: true
title: '[Qt] 关于Qt编译error: undefined reference to `vtable for xxx错误' 
categories:
- C/C++++
tags:
- C/C++++
---

ref: [http://stackoverflow.com/questions/13121310/adding-signal-to-custom-qthread](http://stackoverflow.com/questions/13121310/adding-signal-to-custom-qthread)


Make shure your class is in a different .cpp and .h file that are included in the MOC process generation. undefined reference to `vtable means that the moc cpp file is not generated.


In case you are using Qt Creator and when you compiled your project for the first time, you did not put "Q_OBJECT" in your header file, then the moc cpp file for your (qthread) cpp file was not generated. In this case, simply running "Clean All" and "Rebuild All" after putting "Q_OBJECT" in your header file will not work. You need to go to your build folder to manually delete the Qt generated "Makefile" and run "Rebuild All" or "Build All" again, your error message will be gone.

