# SparkX Subgraph Demo

## 代码

	package com.wzy
	
	import org.apache.spark.graphx.{Edge, Graph, VertexId}
	import org.apache.spark.rdd.RDD
	import org.apache.spark.sql.SparkSession
	
	object subgraphDemo {
	
	  def main(args: Array[String]): Unit = {
	
	    val spark = SparkSession.builder()
	      .appName("spark dataset")
	      .master("spark://192.168.2.123:7077")
	      //本地测试运行需要加这一句话，部署在生产环境则删除
	      .config("spark.jars", "/Users/zheyiwang/IdeaProjects/SparkApps/target/SparkApps-1.0-SNAPSHOT-jar-with-dependencies.jar")
	      .getOrCreate()
	    val sc = spark.sparkContext
	    import spark.implicits._
	
	
	    // Create an RDD for the vertices
	    val users: RDD[(VertexId, (String, String))] =
	      sc.parallelize(Seq((3L, ("rxin", "student")), (7L, ("jgonzal", "postdoc")),
	        (5L, ("franklin", "prof")), (2L, ("istoica", "prof")),
	        (4L, ("peter", "student"))))
	    // Create an RDD for edges
	    val relationships: RDD[Edge[String]] =
	      sc.parallelize(Seq(Edge(3L, 7L, "collab"),    Edge(5L, 3L, "advisor"),
	        Edge(2L, 5L, "colleague"), Edge(5L, 7L, "pi"),
	        Edge(4L, 0L, "student"),   Edge(5L, 0L, "colleague")))
	    // Define a default user in case there are relationship with missing user
	    val defaultUser = ("John Doe", "Missing")
	
	    // Build the initial Graph
	    val graph = Graph(users, relationships, defaultUser)
	    // Notice that there is a user 0 (for which we have no information) connected to users
	    // 4 (peter) and 5 (franklin).
	    graph.triplets.map(
	      triplet => triplet.srcAttr._1 + " is the " + triplet.attr + " of " + triplet.dstAttr._1
	    ).collect.foreach(println(_))
	    // Remove missing vertices as well as the edges to connected to them
	    val validGraph = graph.subgraph(vpred = (id, attr) => attr._2 != "Missing")
	    // The valid subgraph will disconnect users 4 and 5 by removing user 0
	    validGraph.vertices.collect.foreach(println(_))
	    validGraph.triplets.map(
	      triplet => triplet.srcAttr._1 + " is the " + triplet.attr + " of " + triplet.dstAttr._1
	    ).collect.foreach(println(_))
	
	
	  }
	
	}


## 打印

	istoica is the colleague of franklin
	rxin is the collab of jgonzal
	franklin is the advisor of rxin
	peter is the student of John Doe
	franklin is the colleague of John Doe
	franklin is the pi of jgonzal
	
	(4,(peter,student))
	(2,(istoica,prof))
	(3,(rxin,student))
	(7,(jgonzal,postdoc))
	(5,(franklin,prof))
	
	istoica is the colleague of franklin
	rxin is the collab of jgonzal
	franklin is the advisor of rxin
	franklin is the pi of jgonzal
