# ConcurentHashMap

## 背景

在并发编程中，使用HashMap进行put操作会引起死循环，而使用线程安全的HashTable效率又非常低下。所以需要使用ConcurrentHashMap.

HashMap 容器在竞争激烈的并发环境下表现出效率低下的原因是所有访问HashTable的线程都必须竞争同一把锁。ConcurrentHashMap使用锁分段技术，将数据分为一段一段的存储，每一段数据配一把锁。

采用了 CAS + synchronized 来保证并发安全性

![](../Images/1.png)





