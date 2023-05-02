# 简易Demo


## UDF 规则

	spark.udf.register(name, lambda function)


## 代码


	package com.sparrowrecsys
	
	import org.apache.log4j.{Level, Logger}
	import org.apache.spark.sql.{DataFrame, SparkSession}
	
	object SparkUDFDemo {
	  case class Hobbies(name:String,hobbies: String)
	
	  def main(args: Array[String]): Unit = {
	    Logger.getLogger("org").setLevel(Level.ERROR)
	    val spark = SparkSession.builder().master("local[*]").appName("udf").getOrCreate()
	    import spark.implicits._
	
	    val sc = spark.sparkContext
	    val rdd = sc.parallelize(List(("zs,29"),("ls,23")))
	    val df = rdd.map(x=>x.split(","))
	      .map(x=>Hobbies(x(0),x(1))).toDF()
	    df.show()
	
	    /*
	    +----+-------+
	    |name|hobbies|
	    +----+-------+
	    |  zs|     29|
	    |  ls|     23|
	    +----+-------+
	     */
	    //创建视图
	    df.createOrReplaceTempView("df")
	    //定义UDF
	    spark.udf.register("hoby_num",(v:String)=>v.length)
	    spark.udf.register("rib",(v:Int) => v + 1)
	    //使用UDF
	    val frame:DataFrame = spark.sql("select name,hobbies,hoby_num(hobbies) as hobnum from df")
	    frame.show()
	
	    val tt = spark.sql("select rib(hobbies) from df")
	    tt.show()
	    /*
	    +----+-------+------+
	    |name|hobbies|hobnum|
	    +----+-------+------+
	    |  zs|     29|     2|
	    |  ls|     23|     2|
	    +----+-------+------+
	     */
	  }
	}


## 打印

	+----+-------+
	|name|hobbies|
	+----+-------+
	|  zs|     29|
	|  ls|     23|
	+----+-------+
	
	+----+-------+------+
	|name|hobbies|hobnum|
	+----+-------+------+
	|  zs|     29|     2|
	|  ls|     23|     2|
	+----+-------+------+
	
	+-----------------------------+
	|UDF:rib(cast(hobbies as int))|
	+-----------------------------+
	|                           30|
	|                           24|
	+-----------------------------+