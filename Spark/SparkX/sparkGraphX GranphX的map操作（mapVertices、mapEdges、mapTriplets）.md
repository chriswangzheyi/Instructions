# sparkGraphX GranphX的map操作（mapVertices、mapEdges、mapTriplets）

## 概念

* mapVertices 遍历所有的顶点
* mapEdges 遍历所有的边
* mapTriplets 遍历所有的三元组

格式：

	def mapVertices[VD2](map:(VertexId, VD)=> VD2): Graph[VD2, ED]
	def mapEdges[ED2](map: Edge[ED] => ED2): Graph[VD, ED2]
	def mapTriplets[ED2](map: EdgeTriplet[VD, ED] => ED2): Graph[VD, ED2]

## Demo



##  打印

	（1）通过上面的项点数据和边数据创建图对象
	
	(4,(David,42))
	(6,(Fran,50))
	(2,(Bob,27))
	(1,(Alice,28))
	(3,(Charlie,65))
	(5,(Ed,55))
	
	Edge(2,1,7)
	Edge(2,4,2)
	Edge(3,2,4)
	Edge(3,6,3)
	Edge(4,1,1)
	Edge(5,2,2)
	Edge(5,3,8)
	Edge(5,6,3)
	
	在已有图上新建新的图
	(4,(David,84))
	(6,(Fran,100))
	(2,(Bob,54))
	(1,(Alice,56))
	(3,(Charlie,130))
	(5,(Ed,110))
	（2）使用mapEdges函数遍历所有的边，新增加一个属性值然后构建出新的图
	Edge(2,1,(7,100))
	Edge(2,4,(2,100))
	Edge(3,2,(4,100))
	Edge(3,6,(3,100))
	Edge(4,1,(1,100))
	Edge(5,2,(2,100))
	Edge(5,3,(8,100))
	Edge(5,6,(3,100))
	（3）使用mapTriplets函数遍历所有的三元组，新增加一个属性值，然后返回新的图
	Edge(2,1,(7,10))
	Edge(2,4,(2,10))
	Edge(3,2,(4,10))
	Edge(3,6,(3,10))
	Edge(4,1,(1,10))
	Edge(5,2,(2,10))
	Edge(5,3,(8,10))
	Edge(5,6,(3,10))