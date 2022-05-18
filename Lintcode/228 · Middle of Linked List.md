# 228 · Middle of Linked List

## 题目

Find the middle node of a linked list and return it.

Example 1:

	Input:  1->2->3
	Output: 2	
	Explanation: return the middle node.
Example 2:

	Input:  1->2
	Output: 1	
	Explanation: If the length of list is even return the center left one.
	
## 代码（快慢指针）

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
	    @return: a middle node of the linked list
	    """
	    def middle_node(self, head: ListNode) -> ListNode:
	        # write your code here
	        if not head:
	            return head
	
	        slow = fast = head
	        fast = fast.next
	        while fast and fast.next:
	            slow = slow.next
	            fast = fast.next.next
	        return slow
        


	