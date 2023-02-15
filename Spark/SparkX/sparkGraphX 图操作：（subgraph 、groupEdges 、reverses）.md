# sparkGraphX 图操作：（subgraph 、groupEdges 、reverses）

参考：https://blog.csdn.net/Strawberry_595/article/details/105529314

## 解释

### subgraph

返回的对象是一个图，图中包含着的顶点和边分别要满足vpred和epred两个函数。（要注意，顶点和边是完全不同的概念，如果一个边被砍掉了，这个边关联的两个顶点并不会受影响）

要注意，在图里，如果一个顶点没了，其对应的边也就没了，但边没了之后，点不会受影响。

所以，subgraph一般用于：restrict the graph to the vertices and edges of interest或者eliminate broken links.

方法的定义：

	def subgraph(
	
	    epred: EdgeTriplet[VD, ED] => Boolean = (x => true),
	
	    vpred: (VertexId, VD) => Boolean = ((v, d) => true)
	
	): Graph[VD, ED]
	

* epred:edge
* vpred:vertex

### reverse

return a new graph with all edge directions reversed.

###  groupEdges:

groupEdges用于将多重图中的相同的顶点之间的边做合并（除了属性其实没其他可以合并的）

方法的定义：

	def groupEdges(merge: (ED, ED) => ED): Graph[VD, ED]


### mask

mask用于创建一个子图，子图中包含在输入的图中也包含的顶点和边。该方法通常跟subgraph方法一起使用，来根据另一个关联的图来过滤当前的图中展示的数据


## Demo

	package com.wzy
	
	
	import org.apache.spark.graphx.{Edge, Graph, VertexId}
	import org.apache.spark.rdd.RDD
	import org.apache.spark.sql.SparkSession
	
	object MapGraphX2 {
	  def main(args: Array[String]): Unit = {
	
	
	    val spark = SparkSession.builder()
	      .appName("spark dataset")
	      .master("spark://192.168.2.123:7077")
	      //本地测试运行需要加这一句话，部署在生产环境则删除
	      .config("spark.jars", "/Users/zheyiwang/IdeaProjects/SparkApps/target/SparkApps-1.0-SNAPSHOT-jar-with-dependencies.jar")
	      .getOrCreate()
	    val sc = spark.sparkContext
	    import spark.implicits._
	
	
	    //设置users顶点
	    val users: RDD[(VertexId, (String, Int))] =
	      sc.parallelize(Array(
	        (1L, ("Alice", 28)),
	        (2L, ("Bob", 27)),
	        (3L, ("Charlie", 65)),
	        (4L, ("David", 42)),
	        (5L, ("Ed", 55)),
	        (6L, ("Fran", 50))
	      ))
	
	    //设置relationships边
	    val relationships: RDD[Edge[Int]] =
	      sc.parallelize(Array(
	        Edge(2L, 4L, 7),
	        Edge(2L, 4L, 2),
	        Edge(3L, 2L, 4),
	        Edge(3L, 6L, 3),
	        Edge(4L, 1L, 1),
	        Edge(5L, 2L, 2),
	        Edge(5L, 3L, 8),
	        Edge(5L, 6L, 3)
	      ))
	    // 定义默认的作者,以防与不存在的作者有relationship边
	    val defaultUser = ("John Doe", 0)
	
	    println("（1）通过上面的项点数据和边数据创建图对象")
	    // Build the initial Graph
	    val graph: Graph[(String, Int), Int] = Graph(users, relationships,
	      defaultUser)
	    graph.edges.collect.foreach(println(_))
	    graph.vertices.collect.foreach(println(_))
	
	
	
	
	    //设置users顶点
	    val users2: RDD[(VertexId, (String, Int))] =
	      sc.parallelize(Array(
	        (1L, ("Alice2", 28)),
	        (2L, ("Bob2", 27)),
	        (3L, ("Charlie2", 65)),
	        (4L, ("David2", 42))
	      ))
	
	    //设置relationships边
	    val relationships2: RDD[Edge[Int]] =
	      sc.parallelize(Array(
	        Edge(2L, 3L, 7),
	        Edge(2L, 4L, 2),
	        Edge(3L, 2L, 4),
	        Edge(4L, 1L, 1)
	      ))
	    // 定义默认的作者,以防与不存在的作者有relationship边
	    val defaultUser2 = ("Missing", 0)
	
	    println("（1.1）通过上面的项点数据和边数据创建图对象")
	    // Build the initial Graph
	    val graph2: Graph[(String, Int), Int] = Graph(users2, relationships2,
	      defaultUser2)
	    graph2.edges.collect.foreach(println(_))
	    graph2.vertices.collect.foreach(println(_))
	/*    println("（2）对上述的顶点数据和边数据进行修改，再创建一张图2，使得图2中有一些点和边在图1中不存在，然后调用图1的mask方法，传入图2作为参数，观察返回的图3的信息")
	
	    println("mask效果：返回公共子图")
	    val graph3 = graph.mask(graph2)
	    graph3.edges.collect.foreach(println(_))
	    graph3.vertices.collect.foreach(println(_))*/
	
	    println("（3）基于上面的数据进行修改练习subgraph 、groupEdges 、reverses方法")
	
	    println("-----------subgraph")
	    var graph4: Graph[(String, Int), Int] = graph.subgraph(epred = (ed) => !
	      (ed.srcId == 1L || ed.dstId == 2L)
	      , vpred = (id, attr) => id != 3L)
	    graph4.edges.collect.foreach(println(_))
	    graph4.vertices.collect.foreach(println(_))
	
	    println("-----------groupEdges")
	    var graph5: Graph[(String, Int), Int]= graph.groupEdges(merge = (ed1,
	                                                                     ed2) =>
	      (ed1 + ed2 + 100000000))
	    graph5.edges.collect.foreach(println(_))
	    //    graph5.vertices.collect.foreach(println(_))
	
	
	    println("----------reverses")
	    graph2.edges.foreach(println(_))
	    println("-------")
	    graph2.reverse.edges.foreach(println(_))
	
	  }
	}




## 打印

	(1)通过上面的项点数据和边数据创建图对象
	
	Edges:
	
	Edge(2,4,7)
	Edge(2,4,2)
	Edge(3,2,4)
	Edge(3,6,3)
	Edge(4,1,1)
	Edge(5,2,2)
	Edge(5,3,8)
	Edge(5,6,3)
	
	vertices:
	
	(4,(David,42))
	(6,(Fran,50))
	(2,(Bob,27))
	(1,(Alice,28))
	(3,(Charlie,65))
	(5,(Ed,55))

	(1.1)通过上面的项点数据和边数据创建图对象
	
	Edges:
	
	Edge(2,3,7)
	Edge(2,4,2)
	Edge(3,2,4)
	Edge(4,1,1)
	
	vertices:
	
	(1,(Alice2,28))
	(2,(Bob2,27))
	(3,(Charlie2,65))
	(4,(David2,42))

	(3)基于上面的数据进行修改练习subgraph 、groupEdges 、reverses方法

	-----------subgraph
	
	Edge(2,4,7)
	Edge(2,4,2)
	Edge(4,1,1)
	Edge(5,6,3)

	(4,(David,42))
	(6,(Fran,50))
	(2,(Bob,27))
	(1,(Alice,28))
	(5,(Ed,55))
	
	-----------groupEdges
	
	Edge:
	
	Edge(2,4,100000009)
	Edge(3,2,4)
	Edge(3,6,3)
	Edge(4,1,1)
	Edge(5,2,2)
	Edge(5,3,8)
	Edge(5,6,3)
	
	Vertex:
	
	(4,(David,42))
	(6,(Fran,50))
	(2,(Bob,27))
	(1,(Alice,28))
	(3,(Charlie,65))
	(5,(Ed,55))
	
	----------reverses


