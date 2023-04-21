---
author: leon
comments: true
date: 2015-08-27 10:35:20+00:00
layout: post
math: true
title: '[Rails] Ruby & Rails常用技巧集锦' 
categories:
- Ruby
tags:
- Ruby
---


### Rails使用环境变量

Many applications require configuration of settings such as email account credentials or API keys for external services. You can pass local configuration settings to an application using environment variables.

**Gmail Example**

Consider an application that uses Gmail to send email messages. Access to Gmail requires a username and password for access to your Gmail account. In your Rails application, you will need to configure these credentials in the file config/environments/production.rb. A portion of the file might look like this:

```
config.action_mailer.smtp_settings = {
  address: "smtp.gmail.com",
  port: 587,
  domain: "example.com",
  authentication: "plain",
  enable_starttls_auto: true,
  user_name: ENV["GMAIL_USERNAME"],
  password: ENV["GMAIL_PASSWORD"]
}
```

You could “hardcode” your Gmail username and password into the file but that would expose it to everyone who has access to your git repository. Instead use the Ruby variable `ENV["GMAIL_USERNAME"]` to obtain an environment variable. The variable can be used anywhere in a Rails application. Ruby will replace ENV["GMAIL_USERNAME"] with an environment variable.

Let’s consider how to set local environment variables.

### Ruby提取切片

You can get the result you want using collect! or map! to modify the array in-place:

```ruby
# 1. collect
x = %w(hello there world)
x.collect! { |element|
  (element == "hello") ? "hi" : element
}
puts x

# 2. map
x = %w(hello there world)
x.map! { |element|
  if(element == "hello")
      "hi" # change "hello" to "hi"
  else
      element
  end
}
puts x # output: [hi there world]
```

At each iteration, the element is replaced into the array by the value returned by the block.


