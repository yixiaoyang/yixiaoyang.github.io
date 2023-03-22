---
author: leon
comments: true
date: 2023-02-11 21:47+00:00
layout: post
math: true
title: '[go] 一些常用的go patterns'
categories:
- go
tags:
- go
- 编程
- 设计模式
---


## <a name=''></a>单例模式
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

## <a name='-1'></a>耗时记录
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

## <a name='-1'></a>观察者模式
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
## <a name='pipeline'></a>pipeline模式

```go
package patterns

func FanIn(input1, input2 <-chan int) <-chan int {
	c := make(chan int)
	go func() {
		for {
			c <- <-input1
		}
	}()

	go func() {
		for {
			c <- <-input2
		}
	}()
	return c
}

func FanIn2(input1, input2 <-chan int) <-chan int {
	c := make(chan int)
	go func() {
		for {
			select {
			case v := <-input1:
				c <- v
			case v := <-input2:
				c <- v
			}
		}
	}()
	return c
}

func FanOut(input <-chan int, outputs []chan<- int, exitChan <-chan int) {
	for _, output := range outputs {
		go func(out chan<- int) {
			for {
				select {
				case v, ok := <-input:
					{
						if !ok {
							return
						}
						out <- v
					}
				case <-exitChan:
					{
						return
					}
				}
			}
		}(output)
	}
}

```

## <a name='-1'></a>访问者模式
```go
package patterns

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
)

type Visitor func(Shape)
type Shape interface {
	Accept(Visitor)
}

type Circle struct {
	R int
}

type Rectangle struct {
	W int
	H int
}

func (c *Circle) Accept(v Visitor) {
	v(c)
}

func (c *Rectangle) Accept(v Visitor) {
	v(c)
}

func JsonVisitor(s Shape) {
	bytes, err := json.Marshal(s)
	if err == nil {
		fmt.Println(string(bytes))
	}
}

func XmlVisitor(s Shape) {
	bytes, err := xml.Marshal(s)
	if err == nil {
		fmt.Println(string(bytes))
	}
}
```

## <a name='-1'></a>代码仓库

https://github.com/yixiaoyang/go-simplelib.git

## <a name='-1'></a>参考
- https://go.dev/blog/pipelines
- https://refactoringguru.cn/design-patterns/go