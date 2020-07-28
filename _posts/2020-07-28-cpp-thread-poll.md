---
author: leon
comments: true
date: 2020-07-28 07:38+00:00
layout: post
title: '[C++]撸一个线程池'
categories:
- C++
tags:
- C++
---

C++11新增了多线程，实现线程池就比较简单了，要点是bind、函数对象和`condition_variable`。

```cpp
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <iostream>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>
#include <thread>
#include <future>
#include <cstdio>

typedef std::function<void()> task_t;

class ThreadPool
{
public:
    ThreadPool(const int n_threads=16);
    ~ThreadPool();
    bool Shutdown() const;
    void AddTask(const task_t&);
    void Start();
    void Stop();
private:
    void Loop();
    task_t Take();
private:
    std::condition_variable _task_cv;
    std::vector<std::thread> _threads;
    std::mutex _task_mutext;
    bool _shutdown;
    std::queue<task_t> _task_queue;
    int _threads_cnt;
};

#endif // THREADPOOL_H

```

```cpp
#include "ThreadPool.h"
#include <future>
#include <cassert>

ThreadPool::ThreadPool(const int n_threads):
    _shutdown(false),
    _threads_cnt(n_threads)
{
}

ThreadPool::~ThreadPool()
{
    _shutdown = true;
    _task_cv.notify_all();
    for(std::thread& t : _threads){
        if(t.joinable())
            t.join();
    }
}

bool ThreadPool::Shutdown() const
{
    return _shutdown;
}

void ThreadPool::AddTask(const task_t &task)
{
    std::unique_lock<std::mutex> lock(_task_mutext);
    _task_queue.push(task);
    _task_cv.notify_one();
}

void ThreadPool::Start()
{
    assert(_threads.empty());
    _shutdown = false;
    _threads = std::vector<std::thread>(_threads_cnt);
    for(int i = 0; i < _threads_cnt; i++){
        _threads.push_back(std::thread(
            std::bind(&ThreadPool::Loop, this)
        ));
    }
}

void ThreadPool::Stop()
{
    _shutdown = true;
    _task_cv.notify_all();
    for(std::thread& t : _threads){
        if(t.joinable())
            t.join();
    }
}

void ThreadPool::Loop()
{
    while(!_shutdown){
        task_t task = Take();
        if(task){
            task();
        }
    }
}

task_t ThreadPool::Take()
{
    std::unique_lock<std::mutex> lock(_task_mutext);
    while(_task_queue.empty() && (!_shutdown)){
        _task_cv.wait(lock);
    }
    task_t task;
    if((!_task_queue.empty()) && (!_shutdown)){
        task = _task_queue.front();
        _task_queue.pop();
    }
    return task;
}
```

```cpp
#include <iostream>
#include "ThreadPool.h"

ThreadPool pool;
int counter = 0;

void task1(){
    int c = counter++;
    printf("iam task1 id %d\n",c);
    _sleep(1000);
    printf("iam task1 id %d exit\n",c);
}
void task2(){
    int c = counter++;
    printf("iam task2 id %d\n",c);
    _sleep(1500);
    printf("iam task2 id %d exit\n",c);
}
void task3(){
    int c = counter++;
    printf("iam task3 id %d\n",c);
    _sleep(200);
    printf("iam task3 id %d exit\n",c);
}

int main(int , char **)
{
    pool.Start();

    task_t t1(task1);
    task_t t2(task2);
    task_t t3(task3);
    pool.AddTask(t1);
    pool.AddTask(t2);
    pool.AddTask(t3);
    pool.AddTask(t1);
    pool.AddTask(t2);
    pool.AddTask(t3);

    printf("main wait...\n");
    char c = getchar();
    pool.Stop();
    printf("main exit.. getch %c\n",c);
    return 0;
}

```