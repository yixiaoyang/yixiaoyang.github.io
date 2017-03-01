---
author: leon
comments: true
date: 2014-05-08 10:00:05+00:00
layout: post
title: 'Git Server on Centos via SSH' 
categories:
- 工具

tags:
- C
---

### Git Server on Centos via SSH

本wiki为Centos上部署git服务器提供文档。

#### 基本软件安装

<pre>
$ yum install python python-setuptools
$ git clone git://eagain.net/gitosis.git gitosis
$ python setup.py install --record uninstall.txt 
$ su git
$ gitosis-init < /tmp/id_rsa.pub 
$ cd /var/git/
$ ln -s /home/git/repositories/ repo
$ chmod 755 /var/git/repo/gitosis-admin.git/hooks/post-update
$ ssh git@server.holocene-instrument.com
ERROR:gitosis.serve.main:Need SSH_ORIGINAL_COMMAND in environment.
Connection to server.holocene-instrument.com closed.
</pre>

大功告成，说明gitosis生效

#### 添加用户（ssh）

To add new users:
- add a keys/USER.pub file
- authorize them to read/write repositories as needed (or just authorize the group @all)
To create new repositories, just authorize writing to them and push. It's that simple! For example: let's assume your username is jdoe and you want to create a repository myproject. In your clone of gitosis-admin, edit gitosis.conf and add:
<pre>
[group myteam]
members = jdoe
writable = myproject
</pre>

Commit that change and push. Then create the initial commit and push it:

<pre>
mkdir myproject
cd mypyroject
git init
git remote add myserver git@MYSERVER:myproject.git
# do some work, git add and commit files
git push myserver master:refs/heads/master
</pre>

That's it. If you now add others to members, they can use that repository too.

#### 权限修改
提交本地工程到服务器，首先clonegitosis-admin项目到本地
<pre>
$ git clone git@server.holocene-instrument.com:gitosis-admin.git
Cloning into gitosis-admin
remote: Counting objects: 5, done.
remote: Compressing objects: 100% (5/5), done.
remote: Total 5 (delta 0), reused 5 (delta 0)
Receiving objects: 100% (5/5), done.
</pre>
修改查看到gitosis.conf
<pre>
[gitosis]
[group gitosis-admin]
writable = gitosis-admin
members = yuan
[group eontime]
writable = costdb \
           oryx-editor \
           demo/test
members = yuan
</pre>
如上所示，分有2个组类型admin和我们自定义的，你可以随便定义一个组，其实就是个权限集合.writable意思是，这个权限组有哪几个项目的写权限？这里是空格分开，如果太长就换行，例如demo/test的话，就是：

<pre>$ git remote add origin git@gitserver:demo/test.git</pre>

members就是有这个权限的组成员了，通过把有权限的开发者的公钥上传至keydir,最后将gitosis-admin提交至远程

<pre>$ git push remote origin master</pre>
服务端将同步有一个仓库了。


#### 新建git库
服务器端新建git空库

<pre>
$ ssh xxx@server.holocene-instrument.com
  ...
$ su git
$ cd /home/git/repositories
$ ./gitc.sh bone
</pre>
在客户端拷贝新的空库
<pre>$ git clone git@server.holocene-instrument.com:bone.git</pre>
如果拥有读写权限就可以开始使用git工作了。

### Q&A

**(1)ERROR:gitosis.serve.main:Repository read access denied**

原因:gitosis.conf中的members与keydir中的用户名不一致，如gitosis中的members = foo@bar，但keydir中的公密名却叫foo.pub

解决:使keydir的名称与gitosis中members所指的名称一致。 

**(2)git push时提示remote:error:refusing to update checked out branch:refs/heads/master**

原因:这是在服务器创建git新库时使用git init而没有使用git init --bare引起git服务器默认拒绝了push操作

解决:需要进行设置可修改.git/config文件后面添加如下代码
<pre>
[receive]
denyCurrentBranch = ignore
</pre>

**(3)自己写的库创建脚本**

```bash
#!/bin/bash
#
#
# log:
# 	1.	create, 11Jan14


if [ $# -lt 1 ] 
then
	echo "Argument error:$# arguments"
	exit
fi

repo="$1.git"
mkdir $repo
echo "mkdir $repo"

cd $repo
git init --bare
touch Readme.rm
date +"%Y-%m-%d %H:%M:%S" >> Readme.rm
echo "by git." >> Readme.rm


echo "Create repo $repo succced~"
```

**(4)Split subtree as new repo**

参考：[http://stackoverflow.com/questions/359424/detach-subdirectory-into-separate-git-repository/17864475#17864475](http://stackoverflow.com/questions/359424/detach-subdirectory-into-separate-git-repository/17864475#17864475)

**(5)如何恢复误删除的git文件**

直接从本地把文件checkout出来就可以了，用不着从远程服务器上pull下来，因为，所有的历史版本你的本地都有的。 具体做法 git checkout file 同时恢复多个被删除的文件： 
```bash
git ls-files -d | xargs -i git checkout {}
```

- Q1: 如果是不小心执行了git reset，还有办法取消吗？ A:git reflog 查看操作历史，找到之前 HEAD 的 hash 值，然后 `git reset –hard` 到那个 hash 即可。
- Q2: 怎样找回历史版本中删除的文件？ A:先确定需要恢复的文件要恢复成哪一个历史版本(commit)，假设那个版本号是： `commit_id`，那么`git checkout [commit_id] – <path_to_file> `就可以恢复 


### Reference

[http://www.blogjava.net/yuanqixun/archive/2011/05/26/351057.html](http://www.blogjava.net/yuanqixun/archive/2011/05/26/351057.html)
