# 66 · Binary Tree Preorder Traversal

## 题目

Given a binary tree, return the preorder traversal of its nodes' values.

Example 1:

Input:

	binary tree = {1,2,3}
Output:

	[1,2,3]
Explanation:

	      1
	    /   \
	  2       3
It will be serialized as {1,2,3} preorder traversal

Example 2:

Input:

	binary tree = {1,#,2,3}
Output:

	[1,2,3]
Explanation:

     1
       \
        2
       /
      3
It will be serialized as {1,#,2,3} preorder traversal

## 代码

	from typing import (
	    List,
	)
	from lintcode import (
	    TreeNode,
	)
	
	"""
	Definition of TreeNode:
	class TreeNode:
	    def __init__(self, val):
	        self.val = val
	        self.left, self.right = None, None
	"""
	
	class Solution:
	    """
	    @param root: A Tree
	    @return: Preorder in ArrayList which contains node values.
	    """
	    def preorder_traversal(self, root: TreeNode) -> List[int]:
	        # write your code here
	        def preorder(root: TreeNode):
	            if not root:
	                return
	            res.append(root.val)
	            preorder(root.left)
	            preorder(root.right)
	        
	        res = list()
	        preorder(root)
	        return res
	        
