# 67 · Binary Tree Inorder Traversal

## 题目

Given a binary tree, return the inorder traversal of its nodes‘ values.

Example 1:

Input:

	binary tree = {1,2,3}
Output:

	[2,1,3]
Explanation:

	      1
	    /   \
	  2       3
It will be serialized as {1,2,3} inorder traversal

Example 2:

Input:

	binary tree = {1,#,2,3}
Output:

	[1,3,2]
Explanation:

	     1
	       \
	        2
	       /
	      3
It will be serialized as {1,#,2,3} inorder traversal

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
	    @return: Inorder in ArrayList which contains node values.
	    """
	    def inorder_traversal(self, root: TreeNode) -> List[int]:
	        # write your code here
	        res = []
	        def inorder(root):
	          if not root:
	            return
	          inorder(root.left)
	          res.append(root.val)
	          inorder(root.right)
	        inorder(root)
	        return res
