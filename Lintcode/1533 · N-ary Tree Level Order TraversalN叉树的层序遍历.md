# 1533 · N-ary Tree Level Order Traversal

## 题目

Given an n-ary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example, given a 3-ary tree:

![](Images/1.png)

We should return its level order traversal:

	[
	     [1],
	     [3,2,4],
	     [5,6]
	]
	
	
Example 1:

	Input：{1,3,2,4#2#3,5,6#4#5#6}
	Output：[[1],[3,2,4],[5,6]]
	Explanation：Pictured above

Example 2:

	Input：{1,3,2#2#3}
	Output：[[1],[3,2]]
	Explanation：
	          1
		 / \
		3   2