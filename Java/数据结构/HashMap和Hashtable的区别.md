# HashMap和Hashtable的区别

## 不同点

### 父类
* HashTable是继承自Dictionary
* HashMap是继承自AbstractMap类

### 底层数据结构不同

jdk1.7前两者解决哈希冲突的底层都是数组+链表。

但JDK1.8 以后的 HashMap 在解决哈希冲突时有了较大的变化，当链表长度大于阈值（默认为8）时，将链表转化为红黑树，以减少搜索时间。Hashtable 没有变化。

### null值问题

* Hashtable既不支持Null key也不支持Null value
* HashMap中，null可以作为键，这样的键只有一个；可以有一个或多个键所对应的值为null

### 线程安全性

* Hashtable是线程安全的，它的每个方法中都加入了Synchronize方法。
* hashMap不是线程安全的

Hashmap我们一般比较少去用，一般会使用ConcurrentHashmap

### 初始容量
HashMap 初始容量16，每次扩容容量变为2倍
Hashtable 初始容量11，每次扩容会变成2n+1


