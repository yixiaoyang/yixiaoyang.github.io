---
author: leon
comments: true
date: 2013-11-14 16:34:23+00:00
layout: post
slug: ajax-rails%e5%ae%9e%e7%8e%b0%e5%b0%8f%e7%bb%93
title: Ajax on Rails实现小结
wordpress_id: 279
categories:
- Ruby
tags:
- Rails
- Ruby
---

今天终于使用Ajax on Rails（服务器端ajax）实现了的关注/取消关注按钮，中间由于共享模板，出现了好多次的模板变量丢失的情况，花了一番力气重新整合局部视图及其模板才得以最终实现。

### 实现过程
现梳理一下响应过程如下： `follow/unfollow按钮点击` --> `remote_post_form远程异步表单提交` --> `(route...etc)` --> `controller create/destroy relationship控制器数据更新` --> `format.js/html根据请求方式渲染js或者html或者json`等等，（服务器会基于这些数据采取一些行动，并且返回一个HTML片断作为它的响应） --> `create.js.erb渲染js模板` --> `renderfollow_form重新渲染关注区域的显示层html片段` --> `客户端JavaScript(由Rails自动地创建)收到该HTML片断并且使用它更新当前HTML页面指定的部分，经常是一个＜div＞标签的内容`。

### Ajax优缺点
然后谈谈ajax的适用性优缺点吧。

#### Ajax的优点
	
  1. 最大的一点是页面无刷新，在页面内与服务器通信，给用户的体验非常好。
	
  2. 使用异步方式与服务器通信，不需要打断用户的操作，具有更加迅速的响应能力。
	
  
  3.可以把以前一些服务器负担的工作转嫁到客户端，利用客户端闲置的能力来处理，减轻服务器和带宽的负担，节约空间和宽带租用成本。并且减轻服务器的负担，ajax的原则是“ 
`按需取数据`”，可以最大程度的减少冗余请求，和响应对服务器造成的负担。
	
  4. 基于标准化的并被广泛支持的技术，不需要下载插件或者小程序。


#### Ajax的缺点
平时我们大多注意的都是ajax给我们所带来的好处诸如用户体验的提升。而对ajax所带来的缺陷有所忽视。ajax的缺陷都是它先天所产生的。
	
  1. **ajax干掉了back按钮，即对浏览器后退机制的破坏**。后退按钮是一个标准的web站点的重要功能，但是它没法和js进行很好的合作。这是ajax所带来的一个比较严重的问题，因为用户往往是希望能够通过后退来取消前一次操作的。那么对于这个问题有没有办法？答案是肯定的，用过Gmail的知道，Gmail下面采用的ajax技术解决了这个问题，在Gmail下面是可以后退的，但是，它也并不能改变ajax的机制，它只是采用的一个比较笨但是有效的办法，即用户单击后退按钮访问历史记录时，通过创建或使用一个隐藏的IFRAME来重现页面上的变更。（例如，当用户在GoogleMaps中单击后退时，它在一个隐藏的IFRA ME中进行搜索，然后将搜索结果反映到Ajax元素上，以便将应用程序状态恢复到当时的状态）。但是，虽然说这个问题是可以解决的，但是它所带来的开发成本是非常高的，和ajax框架所要求的快速开发是相背离的。这是ajax所带来的一个非常严重的问题。

	
  2. **安全问题**，技术同时也对IT企业带来了新的安全威胁，ajax技术就如同对企业数据建立了一个直接通道。这使得开发者在不经意间会暴露比以前更多的数据和服务器逻辑。ajax的逻辑可以对客户端的安全扫描技术隐藏起来，允许黑客从远端服务器上建立新的攻击。还有ajax也难以避免一些已知的安全弱点，诸如跨站点脚步攻击、SQL注入攻击和基于credentials的安全漏洞等。

	
  3. 对搜索引擎的支持比较弱。

	
  4. 另外，像其他方面的一些问题，比如说违背了url和资源定位的初衷。例如，我给你一个url地址，如果采用了ajax技术，也许你在该url地址下面看到的和我在这个url地址下看到的内容是不同的。这个和资源定位的初衷是相背离的。

	
  5. 一些手持设备（如手机、PDA等）现在还不能很好的支持ajax，比如说我们在手机的浏览器上打开采用ajax技术的网站时，它目前是不支持的，当然，这个问题和我们没太多关系。


最后附上git：[https://github.com/yixiaoyang/ruby/tree/master/rails/rails_project/sample_app](https://github.com/yixiaoyang/ruby/tree/master/rails/rails_project/sample_app)

第一本rails神书`The Ruby on Rails Tutorial`到此算是从头到尾啃得差不多了，略微有点遗憾的是里面的单元测试都没有怎么写。实际开发中比较推崇TDD的开发方式，虽然还不知道他的威力，不过圈子的大牛说这个对软件层次、接口变更以及后期维护以及益处无穷 O_O

放一张前端ajax：

[![ajax](http://cdn1.snapgram.co/imgs/2015/07/20/044.png)](http://cdn1.snapgram.co/imgs/2015/07/20/044.png)
