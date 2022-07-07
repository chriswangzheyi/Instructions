# ArrayList和LinkedList区别

## 一句话区别描述

ArrayList底层是数组，查询快、增删慢；  
LinkedList底层是链表，查询慢、增删快。 

 
## 概念
 
### ArrayList
ArrayList是集合的一种实现，实现了List接口，List接口继承了Collection接口。Collection是所有集合类的父类。ArrayList中的元素有序、可重复、可为空

ArrayList的底层是动态数组，是线程不安全的

###LinkedList
LinkedList底层是双向链表，实现了list接口，LinkedList是非线程安全的，LinkedList元素允许为null，允许重复元素。

LinkedList是基于链表实现的，因此插入删除效率高，查找效率低，也不存在容量不足的问题，所以没有扩容的方法
