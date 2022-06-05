# 225 · Find Node in Linked List

## 题目

Find a node with given value in a linked list. Return null if not exists.

Example 1:

	Input:  1->2->3 and value = 3
	Output: The last node.
	
Example 2:

	Input:  1->2->3 and value = 4
	Output: null

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
	    @param head: the head of linked list.
	    @param val: An integer.
	    @return: a linked node or null.
	    """
	    def find_node(self, head: ListNode, val: int) -> ListNode:
	        # write your code here
	        while head:
	            if head.val == val:
	                return head
	            head = head.next
	        return None
