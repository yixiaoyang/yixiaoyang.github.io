---
author: leon
comments: true
date: 2013-10-26 08:53:30+00:00
layout: post
slug: rails%e7%9a%84session%e5%92%8ccookie%e5%ba%94%e7%94%a8%ef%bc%9a%e7%99%bb%e5%bd%95%e6%b3%a8%e9%94%80%e6%a8%a1%e5%9d%97
title: Rails的session和cookie应用：登录注销模块
wordpress_id: 134
categories:
- Ruby
tags:
- Rails
- Ruby
- Web
- 编程
---

http协议是无状态协议，而使用session可以实现对客户端的会话进行管理。对每一次新的请求，rails应用都会查看session状态，如用户是否登录（匿名访问）等。

本例为实现用户的登录和注销，自定义了Sessions_controller和coockie值，将sessions 作为rails资源指定了 only new、create 和 destroy 操作以实现登录，记住登录和注销功能。

**（1）会话数据初始化：**

在每次User model实例化时都生成一个随机token串，供会话使用，本例使用base64算法工具。

    
    class User < ActiveRecord::Base
      ...
      before_create :create_remember_token
    
      def User.encrypt(token)
        Digest::SHA1.hexdigest(token.to_s)
      end
    
      def User.new_remember_token
        SecureRandom.urlsafe_base64
      end
    
      private
        def create_remember_token
        self.remember_token = User.encrypt(User.new_remember_token)
      end
    end


**（2）用户注册：**

User_controller控制的用户注册操作create完成后，新用户的信息将被存入数据库，此时数据库中user的会话token是没有加密的。

**（3）注册完成，直接登录：**

调用sessions_helper sign_in方法自动登录。用户注册及自动登录代码片段：

    
    class UsersController < ApplicationController
      ...
      # 添加session相关资源module
      protect_from_forgery with: :exception
      include SessionsHelper
    
      def new
        @user = User.new
      end
    
      def create
        @user = User.new(user_params)
        if @user.save
        flash[:success] = "Creat Account Success,Welcome #{@user[:name]}!"
    
        # 创建成功，处理session，定向到用户主界面
        sign_in @user
        redirect_to @user
        else
        render 'new'
      end
    end


**（4）sign in登录方法：**

使用一个惟一的的随机串（使用base64生成保证冲突很小）作为下次会话的token存入浏览器cookie。然后将这个token加密（SHA1）存入数据库。

数据库只保存加密的token是因为，即便整个数据库都泄露了，攻击者也无法使用token登入网站。为了让token更安全，我们计划每次会话都生成不一样的token，这样即使会话被劫持了（攻击者偷取 cookie 伪装成某个用户登录），用户下次登录时前一个会话就会失效（幸灾乐祸的想想CSDN去年泄密门的事情吧你就懂了）。

Session完成攻击（Session Fixation）的攻击者利用的就是利用服务器的session不变机制，借他人之手获得认证和授权，然后冒充他人，利用XSS注入代码诱使用户使用伪造的`session token`。如果应用程序在用户首次访问它时为每一名用户建立一个匿名会话，这时往往就会出现会话固定漏洞。之后用户会在不知不觉中使用了这个没有完成验证的伪造的token访问服务器，服务器将会请求用户验证。然后，一旦用户登录，该会话即升级为通过验证的会话。最初，会话token并未被赋予任何访问权限，但在用户通过验证后，这个token也具有了该用户的访问权限。`_session固定攻击`示例如下所示:

[![session_fixation](http://cdn2.snapgram.co/imgs/2015/07/20/session_fixation.png)](http://cdn2.snapgram.co/imgs/2015/07/20/session_fixation.png)


简而言之，这种攻击的本质在于： **WEB应用在认证后没有改写或者更新session，从而导致了认证前的session还能使用**。

本例中存入浏览器的cookie范例:

[![browser_cookie.png](http://cdn2.snapgram.co/imgs/2015/07/20/027.png)](http://cdn2.snapgram.co/imgs/2015/07/20/027.png)

**（5）下次自动登录：**
下次自动登录的用户就可以从 浏览器cookie 中取出token，加密后查询对比数据库，调用sign in方法自动登录，@ **第（4）步**。

**（6）sign out方法：**

首先经由路由到sessions controller的destroy操作，destroy调用sign out，它只需要作两件事，删除cookie数据、删除用户数据 delete logout操作的session处理示例：

    --- !ruby/hash:ActionController::Parameters
    _method: delete
    authenticity_token: vZwmQCRSmkJbBl8CsSAARTFNVu4PtW58J3dv3qK7oN0=
    controller: sessions
    action: destroy


完成sign in， sign out的完整session_helper实现：
    
    =begin
      如果要自动登入用户，就可以从 cookie 中取出token，加密后查询数据库。
      数据库之所以只保存加密后的token是因为，即便整个数据库都泄露了，攻击者
      也无法使用token登入网站。为了让token更安全，我们计划每次会话都生成不
      一样的token，这样即使会话被劫持了（攻击者偷取 cookie 伪装成某个用户
      登录），用户下次登录时前一个会话就会失效。
    =end
    module SessionsHelper 
      def sign_in(user)
        remember_token = User.new_remember_token
    
        # 将token（base64生成随机串）存入cookie，保证每次获取的token值不相同
        # 因为开发者经常要把 cookie 的失效日期设为 20 年后，所以 Rails 特别提供了 permanent 方法
        # 等效于 ：
        #   cookies[:remember_token] = { value:   remember_token,expires: 20.years.from_now.utc }
        cookies.permanent[:remember_token] = remember_token
    
        #
        # 将sha1加密的token存入数据库，即使黑客获取了浏览器cookie，也无法获取当前
        # Active Record提供update_attribute()方法可以将Model对象的某个属性保存到数据库。
        # 
        user.update_attribute(:remember_token, User.encrypt(remember_token))
        self.current_user = user
      end
    
      def current_user=(user)
        @current_user = user
      end
    
      def current_user
        # 获取加密后的token，查询数据库索引获取对应用户
        remember_token = User.encrypt(cookies[:remember_token])
    
        # 语法：||=（“or equals”）操作符
        # 在第一次调用 current_user 方法时调用 find_by 方法，如果后续再调用的话就
        # 直接返回 @current_user 的值，而不必再查询数据库
        @current_user ||= User.find_by(remember_token: remember_token)
      end
    
      def signed_in?
        !current_user.nil?
      end
    
      def sign_out
        self.current_user = nil
        cookies.delete(:remember_token)
      end
    end


**（7）SSL：进一步的保障会话安全，推荐使用ssl**


_In Rails 3.1 and later, this could be accomplished by always forcing SSL connection in your application config file:_ 

    
    config.force_ssl = true


最终实现效果:

[![sign_in_out](http://blog.apluslogicinc.com/wp-content/uploads/2013/10/029.png)](http://blog.apluslogicinc.com/wp-content/uploads/2013/10/029.png)

【附】git地址 ：[https://github.com/yixiaoyang/ruby.git](https://github.com/yixiaoyang/ruby)






