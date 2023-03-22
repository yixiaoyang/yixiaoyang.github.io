---
author: leon
comments: true
date: 2016-01-06 08:29:20+00:00
layout: post
math: true
title: '[Python] 一个抓取985/211大学院系里教师信息的爬虫小框架' 
categories:
- python

tags:
- python
---

### 简介

目标是尽可能自动抓取目标大学、所需院系的在职教师名单和详细信息（主要是联系方式，简历等），结果生成格式化csv文件，包含内容包括姓名、英文名、职称、邮件、电话、部门、个人网站、电话、研究方向等。最大的问题在于各大学校院系的网站千奇百怪，格式不尽相同，工作量大。

思路：爬虫+解析器+存取器

### 依赖

    python 2.7.10
    beautifulsoup4==4.4.1
    lxml==3.4.4（remove）
    requests==2.8.1
    Ghost.py-0.2.3
    python-imaging
    pytesseract
    tesseract-ocr（图像识别）
    wand(imageMagic的绑定)

### 其他库

    ImageMagic
    media-libs/leptonica
    app-text/tesseract-3.04.00-r2::gentoo（需要自行下载源码编译training工具）

### OCR训练工具

    jTessBoxEditor

### 基本使用

我只是写了一个简单的架子，功能：

1. 解析hao123的211/985名单

2. 数据 《=》csv/json相互转化

3. 按照用户选择解析各个学校、学院的在职员工数据（模糊匹配职称、电子邮件、电话等等基本信息），并抓取简历保存在学院目录。

4. 按照设置规则文件抓取目标内容,规则内容如下：

```python
    # 目标所在的区域的tag和attrs（属性）
    self.zone_tag = ''
    self.zone_attr = {}
    # 要抓取的目标标签tag
    self.tag_name = tag_name or ''
    # 要抓取的目标标签的递归父亲tag列表
    # <span><p>
    #   <a>something</a>
    #  </p></span>
    # 如<a> tag的父亲列表为["p","span"]
    self.parents = []    
    # 1. 目标节点属性
    # {
    #   'class':true, true或者false表示是否含有此标签
    #   'width':'21%'，标签为具体值则表示仅当标签=值时抓取
    #   'class':['class1','class2'], 标签为列表时表示仅当标签值为列表中的值才成立。暂未用到
    # }
    self.attrs = {}
    # 2. 结果筛选，暂未用到
    self.select = []
```

一般来说，找到合适的zone、目标的tag、属性就能找到要找的内容。

解析员工数据时，先编辑对应**学校**下面的json文件，在目标**学院**的json结构里添加抓取规则。然后在**学院**目录下定义MyHandler.py文件，写合适的解析脚本进行解析即可得到想要的数据。
    
由于每个学院的解析模式各不相同，所以需要在每个学院目录下自己实现两个方法：

- handler：将找到的目标转化为员工数据实体
- profile_handler：保存简历正文到文件，模糊匹配员工数据并返回

json中的主要数据结构如下所示：

    "__classname__": "Academy",     表示数据的类型，自动写好的
    "__module__": "models",         表示类的模块，自动写好的
    "departments": {                [*] 找到一个学院的所有教师名单的索引页面，并填写进去,这个页面将用来解析employee（在职员工）
                                    格式为名字：网址
        "全职教员":"http://www.math.pku.edu.cn/static/quanzhijiaoyuan.html"
    },
    departmentsRule结构暂时不用管，没用到
    "departmentsRule": {
        "__classname__": "ParseRule", 
        "__module__": "models", 
        "attrs": {}, 
        "parents": [], 
        "select": [], 
        "tag_name": "", 
        "zone_attr": {}, 
        "zone_tag": ""
    }, 
    "departmentsUrl": "",           没用到
    "done": false,                  没用到
    "employees": [],                没用到，自动生成，不用管
    "eng_name": "www.math.pku.edu.cn", 自动解析生成学院的英文名字，不用管
    "hasDepartments": false,            不用管
    "name": "数学科学学院",               自动解析生成，不用管
    "parser": null, 
    "rule": {                       [*]Rule是你的重点，这是一个规则配置文件
        "__classname__": "ParseRule", 表示数据的类型，自动写好的
        "__module__": "models",       表示类的模块，自动写好的  
        "attrs": {},                  目标tag_name的属性，类似zone_attr
        "parents": [],              [*]这个也是重点，目标的父标签，用于缩小解析范围的
        "select": [],                忽略永不到                              
        "tag_name": "a",            [*]目标tag，这个也是重点    
        "zone_attr": {"id":"main"}, [*]目标tag的属性，这个也是重点，如<a class="link">中的class属性为link，所以规则为{"class":"link"}
        "zone_tag": "div"           [*]目标区域tag，这个也是重点，（标签，如<table>，<html>，<p>都是标签），zone用于缩小网站目标范围，过滤冗余信息
    },
    
    "sname": "math",                  学院简称
    "url": "http://www.math.pku.edu.cn/",    学院主页，自动解析出来的
    "web_engine": "urllib2"             一般用urllib2,对于特定的js网站，需要用selen
                
####菜单

    menu0 = '''
    q. 退出
    p. 打印所有大学名单
    1. 抓取211名单到json文件
    2. 从json文件导入211名单
    3. 构建输出目录
    4. 测试抓取院系导航目录
    5. 自动猜测并分析出学校的院系导航链接
    6. 抓取院系的名称和主页

1. out目录下如果存在`china211.json`则将其大学目录数据导入。如果不存在，可运行`python run.py`选择菜单1从hao123抓取。
2. 编辑`china985.json`，删除不感兴趣的大学。如仅保留985+理工。
3. 运行

```
    $ python run.py
    选择6菜单
    选择学校
    选择院系
    开始解析
```

### 数据结构

    China211
    collges + rule
    academies
    Department
    employees

    
#### Parser

Parser主要有两种，一种是SimpleAParser对于简单的链接+简历url正文的教师队伍介绍页面进行解析。一种是SimpleTableParser对表格式的教师队伍介绍页面进行解析。

    l_parsers = {
        "Parser": Parser,
        "Hao123_211_Parser": Hao123_211_Parser,
        "SimpleAParser": SimpleAParser,
        "AutoAcademyParser": AutoAcademyParser,
        "SimpleTableParser":SimpleTableParser
    }


#### Models

Models模块包含了相关数据的结构，如College，Academy，Employee类结构，同时提供obj2json和json2obj等串行化/实例化接口。

###浏览器和javascript支持

对于一些比较变态的主页（主要是javascript数据），使用selenium调用firefox获取数据

### 反爬虫问题

遇到过比较棘手的将邮件等信息用图片方式进行展示的情况，没办法，只能用的OCR方式进行图像文字识别，可能还需要自己训练数据。

### 其他

1. JSon"陷阱"  
    对于自定义类的json化，需要注意成员必须事先定义好。否则在dump时会出现`循环解析`的问题。
2. BeautifulSoup解析器  
    社区推荐优先使用lxml，主要是速度较快。但是使用时发现有时解析不到目标，改为python自带的"html-parser"。后期将分析原因，或者向社区提交bug
3. 中英文名转换
    主要是中文转英文名，使用了pypinyin库

### 源代码

github地址：https://github.com/yixiaoyang/EduParser/
