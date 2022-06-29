# 134 · LRU缓存策略

##  题目

Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and set.

get(key) Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
set(key, value) Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.
Finally, you need to return the data from each get.

Example1

	Input:
	LRUCache(2)
	set(2, 1)
	set(1, 1)
	get(2)
	set(4, 1)
	get(1)
	get(2)
	Output: [1,-1,1]
	Explanation：
	cache cap is 2，set(2,1)，set(1, 1)，get(2) and return 1，set(4,1) and delete (1,1)，because （1,1）is the least use，get(1) and return -1，get(2) and return 1.
Example 2:

	Input：
	LRUCache(1)
	set(2, 1)
	get(2)
	set(3, 2)
	get(2)
	get(3)
	Output：[1,-1,2]
	Explanation：
	cache cap is 1，set(2,1)，get(2) and return 1，set(3,2) and delete (2,1)，get(2) and return -1，get(3) and return 2.

## 代码

	class DLinkedNode:
	    def __init__(self, key=0, value=0):
	        self.key = key
	        self.value = value
	        self.prev = None
	        self.next = None
	
	
	class LRUCache:
	    """
	    @param: capacity: An integer
	    """
	    def __init__(self, capacity):
	        # do intialization if necessary
	        self.cache = dict()
	        # 使用伪头部和伪尾部节点    
	        self.head = DLinkedNode()
	        self.tail = DLinkedNode()
	        self.head.next = self.tail
	        self.tail.prev = self.head
	        self.capacity = capacity
	        self.size = 0
	
	    """
	    @param: key: An integer
	    @return: An integer
	    """
	    def get(self, key):
	        # write your code here
	        if key not in self.cache:
	            return -1
	        # 如果 key 存在，先通过哈希表定位，再移到头部
	        node = self.cache[key]
	        self.moveToHead(node)
	        return node.value
	
	    """
	    @param: key: An integer
	    @param: value: An integer
	    @return: nothing
	    """
	    def set(self, key, value):
	        # write your code here
	        if key not in self.cache:
	            # 如果 key 不存在，创建一个新的节点
	            node = DLinkedNode(key, value)
	            # 添加进哈希表
	            self.cache[key] = node
	            # 添加至双向链表的头部
	            self.addToHead(node)
	            self.size += 1
	            if self.size > self.capacity:
	                # 如果超出容量，删除双向链表的尾部节点
	                removed = self.removeTail()
	                # 删除哈希表中对应的项
	                self.cache.pop(removed.key)
	                self.size -= 1
	        else:
	            # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
	            node = self.cache[key]
	            node.value = value
	            self.moveToHead(node)
	    
	    def addToHead(self, node):
	        node.prev = self.head
	        node.next = self.head.next
	        self.head.next.prev = node
	        self.head.next = node
	    
	    def removeNode(self, node):
	        node.prev.next = node.next
	        node.next.prev = node.prev
	
	    def moveToHead(self, node):
	        self.removeNode(node)
	        self.addToHead(node)
	
	    def removeTail(self):
	        node = self.tail.prev
	        self.removeNode(node)
	        return node

## 思路

LRU 缓存机制可以通过哈希表辅以双向链表实现，我们用一个哈希表和一个双向链表维护所有在缓存中的键值对。

* 双向链表按照被使用的顺序存储了这些键值对，靠近头部的键值对是最近使用的，而靠近尾部的键值对是最久未使用的。
* 哈希表即为普通的哈希映射（HashMap），通过缓存数据的键映射到其在双向链表中的位置。

