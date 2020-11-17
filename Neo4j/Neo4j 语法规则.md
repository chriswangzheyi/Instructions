# Neo4j 语法规则

参考资料：https://blog.csdn.net/yuanyk1222/article/details/80803458

参考资料2： https://blog.csdn.net/qq_37503890/article/details/101565515?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param


## create 创建节点

	create(节点名称：节点标签{属性名：属性值，属性名：属性值...})

例子：

	create(stu:Student:Player{id:1,name:'yyk',class:132})

## merge 在节点不存在时创建，存在时无操作；

	match(stu:Student) return (stu)
	
	match(stu:Student{id:1}) return (stu.name)
	
	match(stu:Student) where stu.id=1 return (stu)

match.return不能单独使用。

## remove 删除节点的属性

	match(t:Teacher) remove t.name


## set 

### 增加/修改节点属性

	match(t:Teacher) set t.name='yyy' return t

### 为已存在的节点添加标签

	match(t:Teacher) set t:Father return t

## delete 删除节点/关系

	match(t:Teacher) delete t
	
	match(s:Student)-[r]-(t:Teacher) delete r,s,t
	
delete节点时，如果节点之间还有关系会报错


## CASE表达式

计算表达式，并按顺序与WHEN子句进行比较，直到找到匹配项。 如果未找到匹配项，则返回ELSE子句中的表达式。 但是，如果没有ELSE情况且未找到匹配项，则返回null。

	CASE test
	
	 WHEN value THEN result
	
	  [WHEN ...]
	
	  [ELSE default]
	
	END


## 指定路径

	(元素)-[关系]->(元素)


## With

WITH语句将分段的查询部分连接在一起，查询结果从一部分以管道形式传递给另外一部分作为开始点。


###  过滤聚合函数结果

聚合的结果必须要通过WITH语句传递才能进行过滤

	MATCH (david { name: 'Tom Hanks' })--()--(otherPerson)
	WITH otherPerson, count(*) AS foaf
	WHERE foaf > 1
	RETURN otherPerson

### 在collect前对结果排序

可以在将结果传递给collect函数之前对结果进行排序，这样就可以返回排过序的列表。

	MATCH (n)
	WITH n
	ORDER BY n.name DESC LIMIT 3
	RETURN collect(n.name)


###  限制路径搜索的分支

可以限制匹配路径的数量，然后以这些路径为基础再做任何类似的有限制条件的搜索。

	MATCH (n { name: 'Tom Hanks' })--(m)
	WITH m
	ORDER BY m.name DESC LIMIT 1
	MATCH (m)--(o)
	RETURN o.name