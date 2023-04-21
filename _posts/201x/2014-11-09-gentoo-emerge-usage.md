---
author: leon
comments: true
date: 2014-05-08 10:00:05+00:00
layout: post
math: true
title: 'Gentoo emerge使用' 
categories:
- gentoo
tags:
- gentoo

---


### gentoo常用命令

emerg的常用用法

#### 更新系统

1. 更新系统并移除孤立依赖的软件包
<code bash>
# emerge --update --deep --newuse world
# emerge --depclean
# revdep-rebuild (搜索相应缺失的库，并且重新emerge相应的包，需安装gentoolkit，并需运行两次以便确认)
</code>

#### emerge常用参数
<pre>
  -world | system， world范围更广，包含了system，这是两个set，前面不用加--或-
  --p pretend 预览
  --a ask 先予询问
  --c clean 清理系统
  --C unmerge 卸载，与emerge相反
  ---depclean 深度清理，移除与系统无关的包
  --h help 帮助文件
  --v verbose 详细内容
  --s search 查找
  --S searchdesc 从文件名和描述中查找，要慢一些
  --u update 升级软件包
  --D deep 计算整个系统的依赖关系
  --e emptytree 清空依赖树，一般不用，危险命令
  --1 oneshot 一次性安装，不将其信息加入系统目录树
  --o onlydeps 只安装其依赖关系，而不安装软件本身
  --t tree 显示其目录树信息
  --k usepkg 使用二进制包
  --K usepkgonly 只使用二进制包
  --f fetchonly 仅下载安装包
  ---sync 从指定的rsync站点更新portage树，先前所作所有portage树更改均失效
  --N newuse 使用新的USE FLAG，如有必要，需重新编译
  --n noreplace 更新system，但先前安装的软件不予覆盖
</pre>

#### 查找最快的mirror

```
# mirrorselect -s3 -b10 -o -D >> /etc/make.conf
```

此命令对每个mirror均下载100k文件，以此确定最快源，要注意修改/etc/make.conf

#### 系统升级

```
# emerge --sync 或 # emerge-webrsync （更新或称同步portage树）
# emerge --update --deep --newuse world
# emerge --depclean -pv
# revdep-rebuild (需安装gentoolkit)
```
如果使用emerge-webrsync命令进行升级，可先下载最新的portage（以日期命名，而不要用latest命名）至/var/tmp/emerge-webrsync/目录

查看安装XXX的情况，同时列出了使用的 USE 和 LINGUAS
`# emerge -pv XXX`

在world中增加记录，避免自己已手动编译的软件被再次编译
`# emerge -n fcitx （添加fcitx至emerge tree）`

配置文件更新工具
`# etc-update 或 # dispatch-conf`

查询XXX包用了什么USE（需gentoolkit）
`# equery uses XXX`

找到 /bin/ls 所属包
`# qfile /bin/ls`

列出 glibc 包所包含文件
`# qlist glibc`

#### /etc/portage/package.*应用举例 

**package.use**

```
sys-apps/man-pages -nls sys-apps/pciutils -zlib media-libs/freetype bindist app-text/acroread linguas_zh_TW linguas_zh_CN linguas_en
```

作用：
不改变全局USE的同时，微调包的USE。 开始2个是说这2个包不使用相应的 USE，第三个说明要单独在这个包使用这个USE，最后一个是调整 LINGUAS 的，很容易明白。

**package.keywordssys-apps/hdparm ~x86**

作用：
指定相应的包的 KEYWORDS。比如你想 hdparm 包用 ~x86 的版本，而不用 x86 的版本，就用这个来指定。 注意，因为 emerge 的设计，如果你的 make.conf 里边指定了 ~x86的话，你不能反过来通过指定 x86 而 不要 ~x86，只能用 -~x86 来达到目的。 引用 gentoo@freenode 上<kojiro>的话： ”ACCEPT_KEYWORDS is incremental“

**package.mask>sys-devel/libtool-1.5.23**

作用：
屏蔽某个包某个版本，或者某些版本，甚至整个包。
比如 libtool-1.5.23b 在我的系统有问题，那么就屏蔽一下，只用 比 1.5.23 小的。

**package.unmask=net-www/apache-2.2.4 games-arcade/stepmania**
和 mask 一样，不过效果正好相反。我要用 2.2.4 的 apache，但是 portage 把他 mask 了，所以手动 unmask 一下。

往 default runlevel 里边加入 XXX 服务 (add)
`# rc-update -a XXX default`

从 default runlevel 里边删除 XXX 服务 (delete)
`# rc-update -d XXX default`

列出 default runlevel 所有的服务 (show)
`# rc-update -s default`

删除过期的包
`# eclean distfiles (请先 emerge gentoolkit)`

清除emerge过程中产生的临时文件
`# rm -rf /var/tmp/portage/*`

清除所有安装源码包（除非太穷，否则不建议）
`# rm -rf /usr/portage/distfiles/*`
