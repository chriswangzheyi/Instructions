# 35 · Reverse Linked List

## 题目

Example 1:

Input:

	linked list = 1->2->3->null
Output:

	3->2->1->null
Explanation:

Reverse Linked List

Example 2:

Input:

	linked list = 1->2->3->4->null
Output:

	4->3->2->1->null
Explanation:

Reverse Linked List

## 代码

	from lintcode import (
	    ListNode,
	)
	
	"""
	Definition of ListNode:
	class ListNode(object):
	    def __init__(self, val, next=None):
	        self.val = val
	        self.next = next
	"""
	
	class Solution:
	    """
	    @param head: The head of linked list.
	    @param val: An integer.
	    @return: The head of new linked list.
	    """
	    def insert_node(self, head: ListNode, val: int) -> ListNode:
	        # write your code here
	        dummy = ListNode(0,head)
	
	        p = dummy
	        while p.next and p.next.val < val:
	            p = p.next
	        node = ListNode(val, p.next)
	        p.next = node
	        return dummy.next

## 解释

如果Input: head = 1->4->6->8->null, val = 5，

dummy = ListNode(0,head) 得到：

	0->1->4->6->8->null