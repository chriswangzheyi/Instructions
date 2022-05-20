# 1609 · Middle of the Linked List

## 题目

Given a non empty single linked list with head, please return the middle node of the linked list.

If there are two middle nodes, return the second middle node.

Example 1:

	Input: 1->2->3->4->5->null
	Output: 3->4->5->null
Example 2:

	Input: 1->2->3->4->5->6->null
	Output: 4->5->6->null


## 代码 （快慢指针）

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
	    @param head: the head node
	    @return: the middle node
	    """
	    def middle_node(self, head: ListNode) -> ListNode:
	        # write your code here.
	
	        if not head:
	            return head
	
	        slow = fast = head
	
	        while fast and fast.next:
	            slow = slow.next
	            fast = fast.next.next
	        return slow
        
        

### 容易错的点



	while fast.next and fast.next.next
	
	