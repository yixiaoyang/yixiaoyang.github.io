---
author: leon
comments: true
date: 2023-02-11 21:47+00:00
layout: post
title: '[go] 一些常用的go patttern'
categories:
- go
tags:
- go
- 编程
- 设计模式
---

# 单例模式
```go
package patterns

import "sync"

type Singleton struct {
	Data map[string]string
	Id   int
}

var (
	once      sync.Once
	singleton *Singleton
	idCount   int
)

func NewSingleton() *Singleton {
	once.Do(func() {
		idCount += 1
		singleton = &Singleton{
			Data: make(map[string]string),
			Id:   idCount,
		}
	})
	return singleton
}

```

# 耗时记录
```go
package patterns

import (
	"fmt"
	"time"
)

func TimeCost(start time.Time, name string) {
	fmt.Printf("[%v] cost %v\n", name, time.Since(start))
}

func LongTimeWorking() {
	defer TimeCost(time.Now(), "LongTimeWorking")
	time.Sleep(time.Millisecond * 500)
}

```

# 观察者模式
```go
package patterns

import "container/list"

type (
	Event struct {
		Msg string
	}

	Notifier interface {
		Add(Observer)
		Remove(Observer)
		Notify(Event)
	}

	Observer interface {
		OnNotify(Event)
	}
)

type (
	ChatNotifier struct {
		observers map[Observer]struct{}
	}

	ChatObserver struct {
		Name      string
		EventList *list.List
	}

	ChatEvent struct {
		Content string
	}
)

func (notifier *ChatNotifier) Add(observer Observer) {
	notifier.observers[observer] = struct{}{}
}

func (notifier *ChatNotifier) Remove(observer Observer) {
	delete(notifier.observers, observer)
}

func (notifier *ChatNotifier) Notify(e Event) {
	for observer := range notifier.observers {
		observer.OnNotify(e)
	}
}

func (observer *ChatObserver) OnNotify(e Event) {
	observer.EventList.PushBack(e)
}
```
# pipeline模式


# 代码仓库

https://github.com/yixiaoyang/go-simplelib.git

# 参考
- https://go.dev/blog/pipelines