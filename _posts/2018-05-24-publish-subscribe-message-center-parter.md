---
author: leon
comments: true
date: 2018-05-20 12:31:00+00:00
layout: post
math: true
title: '[设计模式]订阅-分发模式实现消息中心组件'
categories:
- 设计模式
tags:
- 设计模式
- C/C++++
---

消息中心组件的应用场景非常多，一个基本的消息中心一般需要支持以下几个关键特性：
- 多事件源
- 多接收端
- 事件过滤
- 跨进程通知

在之前接手的几个基于Qt的项目上，我设计了一个消息中心组件，能够在不同界面、不同线程上实现消息分发。此实现依赖于Qt的signal-slot实现，在非Qt环境下可使用Boost.signal代替。

优点：`消息统一处理，全局通知，防止各个模块相互通信造成的混乱`


### 消息体

标记消息源、类型、目标、消息内容。

```c++
#pragma pack(1)
typedef struct MessageHdrStruct{
    mid_t mid;
    mop_t op;
    unsigned short flags;
    unsigned short type;
    unsigned short from;
    unsigned short to;
    union{
        int asInt;
        unsigned int asUint;
        double asDouble;
        bool asBool;
    }opt;
    short retCode;
}MessageHdr;
#pragma pack()

class Message
{
public:
    MessageHdr mhdr;
    void *content;
    int content_len;
    unsigned int sid;
public:
    Message(){
        this->reset();
    }
    ~Message(){
        char* mem = (char*)content;
        if(mem != NULL){
            delete mem;
            content = NULL;
        }
    }
    unsigned int getId();

    /* Deep copy */
    Message(const Message& msg);
    Message(mid_t id, unsigned int from, int flags,int val, mop_t op=MSG_OP_SET);
private:
    void reset();
};

```

### 订阅者(消息源和接收者)

```c++
#define DEF_HANDLER_FUNC_NAME(mid) on_##mid(Message& )
#define DEF_HANDLER_FUNC(mid)      virtual inline \
                                   STATUS \
                                   DEF_HANDLER_FUNC_NAME(mid){\
                                        return STATUS_OK;\
                                   }
class ISubscriber
{
public:
    ~ISubscriber();
protected:
    unsigned int _id_;
    QString _name_;
public:
    ISubscriber();
    void setId(unsigned int);
    void setName(const QString &);

    bool operator ==(unsigned int);
    virtual QObject* getQobject() = 0;
protected:
    // 订阅者只需要继承ISubscriber，在消息中心注册后实现感兴趣消息的处理函数，
    // 当信号源将信号发送到消息中心后，消息中心过滤完成即依次调用订阅列表中的
    // 处理函数。
    //handlers
    DEF_HANDLER_FUNC(MID_GCONFIG_DEBUG)
    DEF_HANDLER_FUNC(MID_GCONFIG_EXPIRED)
protected slots:
    virtual int handleMsg(Message&) = 0;
};

#define HANDLE_MSG_CASE(mid)    case mid: \
                                    ret = DEF_HANDLER_FUNC_NAME2(mid); \
                                    break
int ISubscriber::handleMsg(Message &msg)
{
    int ret = STATUS_OK;

    /* irnore message not for me */
    if(ISSET(msg.mhdr.flags,MSG_FLAGS_NOTIFY_SELF) &&
            msg.mhdr.from != this->_id_){
        return ret;
    }
    /* ignore message from me */
    if(ISSET(msg.mhdr.flags,MSG_FLAGS_NOTIFY_NOTSELF) &&
            msg.mhdr.from == this->_id_){
        return ret;
    }
    /* ignore message not to unique */
    if(ISSET(msg.mhdr.flags,MSG_FLAGS_NOTIFY_UNIQUE) &&
            msg.mhdr.to != this->_id_){
        return ret;
    }

    switch(msg.mhdr.mid){
    HANDLE_MSG_CASE(MID_GCONFIG_DEBUG);
    HANDLE_MSG_CASE(MID_GCONFIG_EXPIRED);
    default:
        break;
    }

    return ret;
}
```

### 发布者(消息中心)

```c++
class IPublisher : public QObject
{
Q_OBJECT
public:
    virtual int registerFor(ISubscriber *pObserver, int id, QString name);
    virtual int unRegisterFor(ISubscriber *pObserver);

    virtual int post(Message& msg, void* param=NULL);
private:
    virtual int handleMsg(Message& msg, void* param=NULL) = 0;
    virtual void notify(Message &msg);
private:
    QReadWriteLock lock;

public:
    IPublisher();
    ~IPublisher();
signals:
    void sendMsg(Message& msg);
};


int IPublisher::registerFor(ISubscriber *pObserver, int id, QString name)
{
    pObserver->setId (id);
    pObserver->setName(name);
    /*  Qt::ConnectionType type = Qt::AutoConnection */
    QObject::connect(this, SIGNAL(sendMsg(Message&)), pObserver->getQobject (), SLOT(handleMsg(Message&)));
    return STATUS_OK;
}

int IPublisher::unRegisterFor(ISubscriber *pObserver)
{
    QObject::disconnect(this, SIGNAL(sendMsg(Message&)), pObserver->getQobject (), SLOT(handleMsg(Message&)));
    return STATUS_OK;
}

int IPublisher::post(Message &msg, void* param)
{
    int ret = handleMsg (msg,param);
    if(ret != STATUS_OK){
        return ret;
    }

    notify(msg);

    if(!param)
        free(param);

    return ret;
}

void IPublisher::notify(Message &msg)
{
    if(msg.mhdr.mid < MID_MAX){
        if(ISNOTSET(msg.mhdr.flags,MSG_FLAGS_NOTIFY_NONE) &&
           (msg.mhdr.op != MSG_OP_GET)){
            emit sendMsg (msg);
        }
    }
}
```

