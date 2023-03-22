---
author: leon
comments: true
date: 2014-01-21 10:37:25+00:00
layout: post
math: true
title: '[Shell] 个人常用Shell集锦' 
categories:
- Shell
tags:
- Shell
---

shell是Unix系统的一个特色，fuck点和hack点并存：

- fuck点： 需要记住一些复杂的命令，看组织得还算不错的man & help，有一个学习的过程，相比所见所得的视窗真是反人类。代表包括grep、sed、awk。
- hack点： 可以用简单的命令完成复杂的工作，如替换多个文档里的字符串，grep、seed、find再结合正则表达式简直无敌，代表包括grep、sed、awk。

本文将平时用的一些shell集合起来，待不时之需。

### 文本替换

1. 替换所有特定后缀的文件中的字符串，无正则 

    ```bash
    find -type f -name "*.markdown" | xargs sed -i 's/${oldstr}/${newstr}/g'
    ```

### awk
1. awk工作模式
    awk被设计成数据流方式处理，内建很多基本的语法，因此使用起来非常灵活和强大，常用作字符处理。最基本的awk命令模式类似：

    `awk ' BEGIN{  print "start" } pattern { commands } END{ print "end" }  file`
    
    > How it Works? The awk command works in the following manner:
    > 
    > - Execute the statements in the BEGIN { commands } block.
    > - Read one line from the file or stdin, and execute pattern { commands }. Repeat this step until the end of the file is reached.
    > - When the end of the input stream is reached, execute the END { commands } block.

    awk首先调用BEGIN区块，然后逐行处理流，按照相应的匹配模式作相应的处理动作，最后调用END区块。 一个简单的行数计数用法如下：

    `awk 'BEGIN {i=0} {i++} END {print i}' filename`

2. 特殊变量

    |Name|    Usage|
    |----|---------|
    |NR  |It stands for number of records and corresponds to current line number under execution.|
    |$0  |It is a variable that contain the text content of current line under execution.        |
    |$1  |It is a variable that holds the text of the first field.                               |
    |$2  |It is the variable that holds the test of the second field text.
    |NF  |It stands for number of fields and corresponds to number of fields in the current line under execution (Fields are delimited by space).|



### Git

1. git恢复已经删除的文件

直接从本地把文件checkout出来就可以了，用不着从远程服务器上pull下来，因为，所有的历史版本你的本地都有的。 
具体做法 `git checkout file` 同时恢复多个被删除的文件：`git ls-files -d | xargs -i git checkout {}`

在使用Git的过程中，有时可能会有一些误操作 比如：执行`checkout -f` 或 `reset -hard` 或 `branch -d`删除一个分支 结果造成本地（远程）的分支或某些commit丢失 可以通过reflog来进行恢复，前提是丢失的分支或commit信息没有被`git gc`清除 一般情况下，gc对那些无用的object会保留很长时间后才清除的 reflog是git提供的一个内部工具，用于记录对git仓库进行的各种操作,可以使用`git reflog show`或`git log -g`命令来看到所有的操作日志。恢复的过程很简单：通过`git log -g`命令来找到我们需要恢复的信息对应的`commit_id`，可以通过提交的时间和日期来辨别。

一个好的办法是运行：

1. `git log –since="2 weeks ago" – myfile` 可以2个星期期间的myfile历史；
2. `git log –branches="develop"` 可以查看develop的commit 
3. 通过`git branch recover_branch[新分支] commit_id`来建立一个新的分支

这样，我们就把丢失的东西给恢复到了`recover_branch`分支上了。

**Q1:** 如果是不小心执行了`git reset`，还有办法取消吗？ 

**A1:** `git reflog` 查看操作历史，找到之前 HEAD 的 hash 值，然后 `git reset –hard` 到那个 hash 即可。

**Q2:** 怎样找回历史版本中删除的文件？

**A2:** 先确定需要恢复的文件要恢复成哪一个历史版本(commit)，假设那个版本号是： `commit_id`，那么`git checkout [commit_id] – <path_to_file>` 就可以恢复

2. git commit生成patch

可以使用如下两种方式:
`git format-patch -1 commit` ：生成的patch有统计信息和git的版本信息
`git diff commit_previous commit > mypatch.diff`：最原始的diff信息，对于这里的commit_previous(commit之前一个commit），可以使用“commit^”来表示，这样比较方便，不易出错。

3.如何将已有git项目迁移到自己的服务器

可以通过`remote set-url`重设服务器地址，然后将origin push到自己的服务器。
```
git remote add origin/master git@gaoee.net:GaoeeDev/jsoncpp.git
git remote set-url origin git@gaoee.net:GaoeeDev/jsoncpp.git
git push origin master
```

### Diff & Patch

1. diff

    ```bash
    cd your-kernel-tree
    cd ..
    diff -Narup org-kernel-source your-kernel-tree > kernel.patch
    ```
-x 过滤掉所比较目录中一些不想比较的文件类型，可以使用其他的pattern。实际上如果需要过滤的文件类型比较多的时候，使用-x这个选项就有点麻烦了，查看了文档之后，diff提供了更加方便的参数过滤文件。 -X excludefile 忽略在excludefile中的文件类型，注意每种文件占一行 pallying patches

2. patch

    ```bash
    cd kernel-source
    patch -p1 < ../kernel.patch
    patch -R -p1 < ../kernel.patch
    ```
   
### eix 

gentoo下经常使用的软件包查询管理工具

```
eix_tag  
update-eix-remote update 可以查到未下载的overlay里的东西  
equery uses XXX    查询XXX包带的USE标志  
equery hasuse X    查询使用X这个USE标志的包  
eix -2 -I --only-names 查slot  
eix -1 -I --only-names   
eix -U(--use) ithread 查询USE为ithread的包   
eix -v(--verbose)    详细查询  

eix -l( --versionlines )  每个版本都以行列出  
eix -c(--compact) 只列出一些信息  

eix -d , --dup-packages  只匹配duplicated的包 如果sys-foo/bar同时存在不同的overlay里面(包括官方portage)  
eix -D, --dup-versions  同时在存不同的版本,类似-d  
eix -P,--provide i.e "virtual/blackbox"  
eix --only-names 只列出名字  
eix -I(--installed) 列出已完装的  
eix -i(--multi-installed)  
eix -u(--upgrade, --upgrade+, --upgrade-) 升级  
eix --stable    至少有一个是稳定版的包  
eix --system 列出是system的包  
eix -O, --overlay             到少匹配一个包版本在Overaly里的包  
eix --in-overlay overlay_name 列出在overlay_name里的包(注：不能加overlay_name不知为何)  
eix --only-in-overlay overlay_name  
eix -J(--installed-overlay)  安装了overaly的包  
eix --installed-from-overlay overlay  
eix -s, --name 默认以名字查询  
eix -S, --description 以描述查询  
eix -C, --category  i.e. "app-portage"  
eix -A, --category-name  i.e. "app-portage/eix"  
eix -H, --homepage   i.e "http://xxx"  
eix -L, --license    i.e "GPL-2"  
eix --installed-with-use  安装包带use参数的  
eix --installed-without-use  
eix -e, --exact  直接查完整包名 如 eix -e gcc 查出只是gcc的包  
eix -f, --fuzzy 模糊查找  
eix -p, --pattern   
eix -r, --regex 正规表达式  
eix -I -J  列出已安装的overlay的包  
eix --fetch  列出最后一个版本是需要自己手动下载的包  
eix --mirror 列出最后一个版本是 !m 的包   
eix --stable 列出最后一个版本为stable的包  
eix --upgrade, --upgrade+, --upgrade- 最后一个版本为可升级的或是降级+ - 表示LOCAL_PORTAGE_CONFIG的真和假  
eix --testing, --testing+,--testing-  
eix --non-masked, --non-masked+, --non-masked-  
eix --system, --system+, --system-  
eix -O, --overlay 只列出包最后一个版本在overlay，无论是否安装，注意跟-J的区别  
eix -T, --test-obsolete 测试陈旧的包  
eix -l, --pipe   
eix -!, --not    
```

非常有用的FORMAT

   Application:  
   FORMAT='{downgrade}%{FORMAT_COMPACT}{}' eix -I  
   This  will  print  all installed packages for which there are downgrade recommendations.  Note that  
   the compact format is used to output the packages: We  cannot  use  FORMAT='{downgrade}%{FORMAT}{}'  
   because  this  would be a self-reference.  However, if you want to use the default FORMAT layout to  
   output the packages, we can use FORMAT_COMPACT to wrap around FORMAT:  

   FORMAT_COMPACT='{downgrade}%{FORMAT}{}' eix -cI  
   This is as above, but the matching packages will be printed with the default (non-compact)  format.  
   The  option  -c  is  needed  so  that eix will use our FORMAT_COMPACT variable as the format string  
   (which we "misused" as a "wrapper" for FORMAT in this example).  

