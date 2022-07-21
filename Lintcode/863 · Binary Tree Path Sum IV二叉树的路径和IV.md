# 863 · Binary Tree Path Sum IV

If the depth of a tree is smaller than 5, this tree can be represented by a list of three-digits integers.

For each integer in this list:

The hundreds digit represents the depth D of this node, 1 <= D <= 4.
The tens digit represents the position P of this node in the level it belongs to, 1 <= P <= 8. The position is the same as that in a full binary tree.
The units digit represents the value V of this node, 0 <= V <= 9.
Given a list of ascending three-digits integers representing a binary with the depth smaller than 5. You need to return the sum of all paths from the root towards the leaves.


如果一个树的深度小于5，那么这个树可以用一个三位整数的列表表示。

对于列表中的每一个整数：

百位数表示当前节点的深度D，1 <= D <= 4。
十位数表示当前节点在当前层的位置P，1 <= P <= 8。这个位置相当于是在它在满二叉树中的位置。
个位数表示当前节点的值V，0 <= V <= 9。
给定一个升序的三位整数的列表，它表示一个深度小于5的二叉树。你需要返回从根结点到叶子节点的所有路径的和。


Example 1:

	Input: [113, 215, 221]
	Output: 12
	Explanation:
	  The tree that the list represents is:
	    3
	   / \
	  5   1
	  The path sum is (3 + 5) + (3 + 1) = 12.
	  
Example 2:

	Input: [113, 221]
	Output: 4
	Explanation:
	  The tree that the list represents is:
	    3
	     \
	      1
	  The path sum is 3 + 1 = 4.