# SparkX outerjoinvertices

## 概念

outerjoinvertices：
这个操作会把关联上的定点的属性给重新赋值,所以这样join的话就有点leftjoin的感觉

## 源码

	def outerJoinVertices[U, VD2](other: RDD[(VertexID, U)])
	      (mapFunc: (VertexID, VD, Option[U]) => VD2)
	    : Graph[VD2, ED]
	    
mapFunc的几种写法：

	1）val inputGraph: Graph[Int, String] =
	  graph.outerJoinVertices(graph.outDegrees)((vid, _, degOpt) => degOpt.getOrElse(0))
	
	2）val degreeGraph = graph.outerJoinVertices(outDegrees) { (id, oldAttr, outDegOpt) =>
	  outDegOpt match {
	    case Some(outDeg) => outDeg
	    case None => 0 // No outDegree means zero outDegree
	  }
	}
	
	3）val rank_cc = cc.outerJoinVertices(pagerankGraph.vertices) {
	  case (vid, cc, Some(pr)) => (pr, cc)
	  case (vid, cc, None) => (0.0, cc)
	}


## 代码

	package com.wzy
	
	import org.apache.spark.graphx.{Edge, Graph, VertexId}
	import org.apache.spark.rdd.RDD
	import org.apache.spark.sql.SparkSession
	
	object Outerdemo {
	
	  def main(args: Array[String]): Unit = {
	
	    val spark = SparkSession.builder()
	      .appName("people management")
	      .master("spark://192.168.2.123:7077")
	      //本地测试运行需要加这一句话，部署在生产环境则删除
	      .config("spark.jars", "/Users/zheyiwang/IdeaProjects/SparkApps/target/SparkApps-1.0-SNAPSHOT-jar-with-dependencies.jar")
	      .getOrCreate()
	    val sc = spark.sparkContext
	    import spark.implicits._
	    //创建点RDD
	    val usersVertices: RDD[(VertexId, (String, String))] = sc.parallelize(Array((1L, ("Spark", "scala")),
	      (2L, ("Hadoop", "java")), (3L, ("Kafka", "scala")), (4L, ("Zookeeper", "Java "))))
	
	    //创建边RDD
	    val usersEdges: RDD[Edge[String]] = sc.parallelize(Array(Edge(2L, 1L, "study"), Edge(3L, 2L, "train"),
	      Edge(1L, 2L, "exercise"), Edge(4L, 1L, "None")))
	
	    val salaryVertices: RDD[(VertexId, (String, Long))] = sc.parallelize(Array((1L, ("Spark", 30L)), (2L, ("Hadoop", 15L)),
	      (3L, ("Kafka", 10L)), (5L, ("parameter server", 40L))))
	
	    val salaryEdges: RDD[Edge[String]] = sc.parallelize(Array(Edge(2L, 1L, "study"), Edge(3L, 2L, "train"), Edge(1L, 2L, "exercise"),
	      Edge(5L, 1L, "None")))
	
	    //构造Graph
	    val graph = Graph(usersVertices, usersEdges)
	    val graph1 = Graph(salaryVertices, salaryEdges)
	
	    println("graph vertices:")
	    graph.vertices.collect().foreach(println(_))
	
	    println("graph1 vertices:")
	    graph1.vertices.collect().foreach(println(_))
	
	    //outerJoinVertices操作,
	    val joinGraph = graph.outerJoinVertices(graph1.vertices) { (id, attr, deps) =>
	      deps match {
	        case Some(deps) => deps
	        case None => 0
	      }
	    }
	    joinGraph.vertices.collect.foreach(println)
	    sc.stop()
	  }
	}


## 打印


	graph vertices:
	(4,(Zookeeper,Java ))
	(2,(Hadoop,java))
	(1,(Spark,scala))
	(3,(Kafka,scala))
	
	graph1 vertices:
	(2,(Hadoop,15))
	(1,(Spark,30))
	(3,(Kafka,10))
	(5,(parameter server,40))
	
	
	(4,0)
	(2,(Hadoop,15))
	(1,(Spark,30))
	(3,(Kafka,10))