# 137 · Clone Graph

## 题目

Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors. Nodes are labeled uniquely.

You need to return a deep copied graph, which has the same structure as the original graph, and any changes to the new graph will not have any effect on the original graph.

Example1

	Input:
	{1,2,4#2,1,4#4,1,2}
	Output: 
	{1,2,4#2,1,4#4,1,2}
	Explanation:
	1------2  
	 \     |  
	  \    |  
	   \   |  
	    \  |  
	      4   
	Nodes are separated by '#'
	1,2,4indicates  a node label = 1, neighbors = [2,4]
	2,1,4 indicates a node label = 2, neighbors = [1,4]
	4,1,2 indicates a node label = 4, neighbors = [1,2]
	
## 代码 （BFS）

	from lintcode import (
	    UndirectedGraphNode,
	)
	
	"""
	Definition for a UndirectedGraphNode:
	class UndirectedGraphNode:
	    def __init__(self, label):
	        self.label = label
	        self.neighbors = []
	"""
	
	class Solution:
	    """
	    @param node: A undirected graph node
	    @return: A undirected graph node
	    """
	    def clone_graph(self, node: UndirectedGraphNode) -> UndirectedGraphNode:
	        # write your code here
	        from collections import deque
	
	        if not node:
	            return node
	        visited = {}
	        
	        # 将题目给定的节点添加到队列
	        queue = deque([node])
	        # 克隆第一个节点并存储到哈希表中
	        visited[node] = UndirectedGraphNode(node.label)
	        # 广度优先搜索
	        while queue:
	            # 取出队列的头节点
	            n = queue.popleft()
	            # 遍历该节点的邻居
	            for neighbor in n.neighbors:
	                if neighbor not in visited:
	                    # 如果没有被访问过，就克隆并存储在哈希表中
	                    visited[neighbor] = UndirectedGraphNode(neighbor.label)
	                    # 将邻居节点加入队列中
	                    queue.append(neighbor)
	                # 更新当前节点的邻居列表
	                visited[n].neighbors.append(visited[neighbor])
	
	        return visited[node]