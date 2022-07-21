# 68 · Binary Tree Postorder Traversal

## 题目

Given a binary tree, return the postorder traversal of its nodes’ values.

Example 1:

Input:

	binary tree = {1,2,3}
Output:

	[2,3,1]
Explanation:

	      1
	    /   \
	  2       3
It will be serialized to {1,2,3} followed by post-order traversal

Example 2:

Input:

	binary tree = {1,#,2,3}
Output:

	[3,2,1]
Explanation:

	     1
	       \
	        2
	       /
	      3
It will be serialized to {1,#,2,3} followed by post-order traversal

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
	    @return: Postorder in ArrayList which contains node values.
	    """
	    def postorder_traversal(self, root: TreeNode) -> List[int]:
	        # write your code here
	        def postorder(root: TreeNode):
	            if not root:
	                return
	            postorder(root.left)
	            postorder(root.right)
	            res.append(root.val)
	        
	        res = list()
	        postorder(root)
	        return res


## 解释

后序遍历：按照访问左子树——右子树——根节点的方式遍历这棵树，而在访问左子树或者右子树的时候，我们按照同样的方式遍历，直到遍历完整棵树。因此整个遍历过程天然具有递归的性质，我们可以直接用递归函数来模拟这一过程。

定义 postorder(root) 表示当前遍历到 root 节点的答案。按照定义，我们只要递归调用 postorder(root->left) 来遍历 root 节点的左子树，然后递归调用 postorder(root->right) 来遍历 root 节点的右子树，最后将 root 节点的值加入答案即可，递归终止的条件为碰到空节点。

