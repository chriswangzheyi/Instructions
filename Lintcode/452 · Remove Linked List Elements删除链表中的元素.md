# 452 · Remove Linked List Elements

Remove all elements from a linked list of integers that have value val.

Example 1:

	Input: head = 1->2->3->3->4->5->3->null, val = 3
	Output: 1->2->4->5->null
Example 2:

	Input: head = 1->1->null, val = 1
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
	    @param head: a ListNode
	    @param val: An integer
	    @return: a ListNode
	    """
	    def remove_elements(self, head: ListNode, val: int) -> ListNode:
	        # write your code here
	        dummy = ListNode(-1, head)
	        p = dummy
	        
	        while p.next:
	            if p.next.val == val:
	                p.next = p.next.next
	            else:
	               p = p.next
	        return dummy.next

