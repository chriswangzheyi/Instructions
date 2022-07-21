# 174 · Remove Nth Node From End of List

## 题目
Given a linked list, remove the nth node from the end of list and return its head.

Example 1:

		Input: list = 1->2->3->4->5->null， n = 2
		Output: 1->2->3->5->null


Example 2:

	Input:  list = 5->4->3->2->1->null, n = 2
	Output: 5->4->3->1->null

## 代码 （双指针）

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
	    @param head: The first node of linked list.
	    @param n: An integer
	    @return: The head of linked list.
	    """
	    def remove_nth_from_end(self, head: ListNode, n: int) -> ListNode:
	        # write your code here
	        dummy = ListNode(0, head)
	        first = head
	        second = dummy
	        for i in range(n):
	            first = first.next
	
	        while first:
	            first = first.next
	            second = second.next
	        
	        second.next = second.next.next
	        return dummy.next    