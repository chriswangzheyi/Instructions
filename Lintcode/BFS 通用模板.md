# BFS 通用模板

## 代码

	queue - collections.deque([node])
	distance = {node : 0}
	
	while queue:
		node = queue.popleft()
		
		for neighbor in node.get_neighbors():
			if neighbor in distance:
				continue
			distance[neighbor] = distance[node] + 1
			queue.append(neighbor)

## 时间复杂度

	N 个点， M条边， 图上BFS时间复杂度 = O（N + M）