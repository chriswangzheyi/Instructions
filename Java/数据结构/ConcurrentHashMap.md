# ConcurrentHashMap

参考：https://www.jianshu.com/p/5dbaa6707017

## 简介

ConcurrentHashMap是一个线程安全并且高效的Hashmap实现。

![](../Images/3.png)

和 1.8 HashMap 结构类似，当链表节点数超过指定阈值的话，也是会转换成红黑树的，大体结构也是一样的。

## 如何实现线程安全

抛弃了原有的Segment 分段锁，而采用了 CAS + synchronized 来保证并发安全性。至于如何实现，那我继续看一下put方法逻辑

## HashMap、Hashtable、ConccurentHashMap三者的区别

HashMap线程不安全，数组+链表+红黑树

Hashtable线程安全，锁住整个对象，数组+链表

ConccurentHashMap线程安全，CAS+同步锁，数组+链表+红黑树

HashMap的key，value均可为null，其他两个不行。

## 在JDK1.7和JDK1.8中的区别

在JDK1.8主要设计上的改进有以下几点:

1、不采用segment而采用node，锁住node来实现减小锁粒度。

2、设计了MOVED状态 当resize的中过程中 线程2还在put数据，线程2会帮助resize。

3、使用3个CAS操作来确保node的一些操作的原子性，这种方式代替了锁。

4、sizeCtl的不同值来代表不同含义，起到了控制的作用。
采用synchronized而不是ReentrantLock



