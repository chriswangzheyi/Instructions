# agg 函数的用法

## 说明
agg函数经常与groupBy函数一起使用，起到分类聚合的作用；

如果单独使用则对整体进行聚合；

## 代码

	package com.dt.spark.Test
	 
	import org.apache.spark.sql.{DataFrame, SparkSession}
	 
	object AggTest {
	  case class Student(classId:Int,name:String,gender:String,age:Int)
	  def main(args: Array[String]): Unit = {
	    val spark = SparkSession.builder().master("local[*]").appName("testAgg").getOrCreate()
	    import spark.implicits._
	    val sc = spark.sparkContext
	    sc.setLogLevel("WARN")
	 
	    val stuDF: DataFrame = Seq(
	      Student(1001, "zhangsan", "F", 20),
	      Student(1002, "lisi", "M", 16),
	      Student(1003, "wangwu", "M", 21),
	      Student(1004, "zhaoliu", "F", 21),
	      Student(1004, "zhouqi", "M", 22),
	      Student(1001, "qianba", "M", 19),
	      Student(1003, "liuliu", "F", 23)
	    ).toDF()
	 
	    import org.apache.spark.sql.functions._
	 
	    stuDF.groupBy("gender").agg(max("age"),min("age"),avg("age"),count("classId")).show()
	    //同样也可以这样写
	    //stuDF.groupBy("gender").agg("age"->"max","age"->"min","age"->"avg","id"->"count").show()
	 
	    stuDF.agg(max("age"),min("age"),avg("age"),count("classId")).show()
	 
	    stuDF.groupBy("classId","gender").agg(max("age"),min("age"),avg("age"),count("classId")).orderBy("classId").show()
	  }
	}

## 结果输出

	+------+--------+--------+------------------+--------------+
	|gender|max(age)|min(age)|          avg(age)|count(classId)|
	+------+--------+--------+------------------+--------------+
	|     F|      23|      20|21.333333333333332|             3|
	|     M|      22|      16|              19.5|             4|
	+------+--------+--------+------------------+--------------+
	 
	+--------+--------+------------------+--------------+
	|max(age)|min(age)|          avg(age)|count(classId)|
	+--------+--------+------------------+--------------+
	|      23|      16|20.285714285714285|             7|
	+--------+--------+------------------+--------------+
	 
	+-------+------+--------+--------+--------+--------------+
	|classId|gender|max(age)|min(age)|avg(age)|count(classId)|
	+-------+------+--------+--------+--------+--------------+
	|   1001|     F|      20|      20|    20.0|             1|
	|   1001|     M|      19|      19|    19.0|             1|
	|   1002|     M|      16|      16|    16.0|             1|
	|   1003|     M|      21|      21|    21.0|             1|
	|   1003|     F|      23|      23|    23.0|             1|
	|   1004|     F|      21|      21|    21.0|             1|
	|   1004|     M|      22|      22|    22.0|             1|
	+-------+------+--------+--------+--------+--------------+