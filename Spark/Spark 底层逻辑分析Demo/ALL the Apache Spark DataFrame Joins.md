# ALL the Apache Spark DataFrame Joins

参考：https://www.youtube.com/watch?v=CKWtJFvyk4w&list=RDCMUCRS4DvO9X7qaqVYUW2_dwOw&index=2


## 原始数据

### kids

	+---+-------+----+
	| Id|   Name|Team|	
	+---+-------+----+
	| 40|   Mary|   1|  
	| 41|   Jane|   3|   
	| 42|  David|   3|   
	| 43| Angela|   2|   
	| 44|Charlie|   1|    
	| 45|  Jimmy|   2|  
	| 46| Lonely|   7|    
	+---+-------+----+


### team 

	+-------+---------------------+
	| TeamId|        TeamName     |
	+-------+---------------------+
	|   1   |    The Inviccibles  |
	|   2   |      Dog Lovers     |
	|   3   |      Rockstarts     |
	|   4   |The Non-Existent Team|
	+-------+---------------------+



## Joins 结果

### inner join

	+---+-------+----+------+---------------+
	| Id|   Name|Team|TeamId|       TeamName|
	+---+-------+----+------+---------------+
	| 44|Charlie|   1|     1|The Inviccibles|
	| 40|   Mary|   1|     1|The Inviccibles|
	| 41|   Jane|   3|     3|     Rockstarts|
	| 42|  David|   3|     3|     Rockstarts|
	| 43| Angela|   2|     2|     Dog Lovers|
	| 45|  Jimmy|   2|     2|     Dog Lovers|
	+---+-------+----+------+---------------+
	
### left outer joins

	+---+-------+----+------+---------------+
	| Id|   Name|Team|TeamId|       TeamName|
	+---+-------+----+------+---------------+
	| 44|Charlie|   1|     1|The Inviccibles|
	| 40|   Mary|   1|     1|The Inviccibles|
	| 41|   Jane|   3|     3|     Rockstarts|
	| 42|  David|   3|     3|     Rockstarts|
	| 46| Lonely|   7|  null|           null|
	| 43| Angela|   2|     2|     Dog Lovers|
	| 45|  Jimmy|   2|     2|     Dog Lovers|
	+---+-------+----+------+---------------+
	
###  right outer joins

	+----+-------+----+------+--------------------+
	|  Id|   Name|Team|TeamId|            TeamName|
	+----+-------+----+------+--------------------+
	|  44|Charlie|   1|     1|     The Inviccibles|
	|  40|   Mary|   1|     1|     The Inviccibles|
	|  41|   Jane|   3|     3|          Rockstarts|
	|  42|  David|   3|     3|          Rockstarts|
	|null|   null|null|     4|The Non-Existent ...|
	|  43| Angela|   2|     2|          Dog Lovers|
	|  45|  Jimmy|   2|     2|          Dog Lovers|
	+----+-------+----+------+--------------------+

### full outer joins

	+----+-------+----+------+--------------------+
	|  Id|   Name|Team|TeamId|            TeamName|
	+----+-------+----+------+--------------------+
	|  40|   Mary|   1|     1|     The Inviccibles|
	|  44|Charlie|   1|     1|     The Inviccibles|
	|  41|   Jane|   3|     3|          Rockstarts|
	|  42|  David|   3|     3|          Rockstarts|
	|null|   null|null|     4|The Non-Existent ...|
	|  46| Lonely|   7|  null|                null|
	|  43| Angela|   2|     2|          Dog Lovers|
	|  45|  Jimmy|   2|     2|          Dog Lovers|
	+----+-------+----+------+--------------------+


### semi joins

	+---+-------+----+
	| Id|   Name|Team|
	+---+-------+----+
	| 40|   Mary|   1|
	| 44|Charlie|   1|
	| 41|   Jane|   3|
	| 42|  David|   3|
	| 43| Angela|   2|
	| 45|  Jimmy|   2|
	+---+-------+----+
	
### anti join

	+---+------+----+
	| Id|  Name|Team|
	+---+------+----+
	| 46|Lonely|   7|
	+---+------+----+
	
### cross join

	+---+-------+----+------+--------------------+
	| Id|   Name|Team|TeamId|            TeamName|
	+---+-------+----+------+--------------------+
	| 40|   Mary|   1|     1|     The Inviccibles|
	| 41|   Jane|   3|     1|     The Inviccibles|
	| 42|  David|   3|     1|     The Inviccibles|
	| 40|   Mary|   1|     2|          Dog Lovers|
	| 41|   Jane|   3|     2|          Dog Lovers|
	| 42|  David|   3|     2|          Dog Lovers|
	| 40|   Mary|   1|     3|          Rockstarts|
	| 40|   Mary|   1|     4|The Non-Existent ...|
	| 41|   Jane|   3|     3|          Rockstarts|
	| 41|   Jane|   3|     4|The Non-Existent ...|
	| 42|  David|   3|     3|          Rockstarts|
	| 42|  David|   3|     4|The Non-Existent ...|
	| 43| Angela|   2|     1|     The Inviccibles|
	| 44|Charlie|   1|     1|     The Inviccibles|
	| 45|  Jimmy|   2|     1|     The Inviccibles|
	| 46| Lonely|   7|     1|     The Inviccibles|
	| 43| Angela|   2|     2|          Dog Lovers|
	| 44|Charlie|   1|     2|          Dog Lovers|
	| 45|  Jimmy|   2|     2|          Dog Lovers|
	| 46| Lonely|   7|     2|          Dog Lovers|
	+---+-------+----+------+--------------------+
	

## 代码


	package com.wzy
	
	import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
	import org.apache.spark.sql.{Row, SparkSession}
	
	
	object AllTheJoinsDemo {
	
	  val spark =SparkSession.builder()
	    .appName("Repartion and coalesce")
	    .master("spark://192.168.2.113:7077")
	    .getOrCreate()
	
	  val sc = spark.sparkContext
	
	  val kids = sc.parallelize(List(
	      Row(40, "Mary",1),
	      Row(41, "Jane",3),
	      Row(42, "David",3),
	      Row(43, "Angela",2),
	      Row(44, "Charlie",1),
	      Row(45, "Jimmy",2),
	      Row(46, "Lonely",7),
	  ))
	
	  val kidsSchema = StructType(Array(
	    StructField("Id",IntegerType),
	    StructField("Name",StringType),
	    StructField("Team",IntegerType),
	  ))
	
	  val kidsDF = spark.createDataFrame(kids, kidsSchema)
	
	  val teams = sc.parallelize(List(
	    Row(1, "The Inviccibles"),
	    Row(2, "Dog Lovers"),
	    Row(3, "Rockstarts"),
	    Row(4, "The Non-Existent Team"),
	  ))
	
	  val teamsSchema =StructType(Array(
	    StructField("TeamId",IntegerType),
	    StructField("TeamName",StringType),
	  ))
	
	  val teamDF = spark.createDataFrame(teams, teamsSchema)
	
	  //inner join
	  val joinCondition = kidsDF.col("Team") === teamDF.col("TeamId")
	  val kidsTeamsDF = kidsDF.join(teamDF, joinCondition,"inner")
	  kidsTeamsDF.show()
	
	  // left outer joins
	  val allKidsTeamsDF = kidsDF.join(teamDF, joinCondition,"left_outer")
	  allKidsTeamsDF.show()
	
	  // right outer joins
	  val allTeamsKidsDF = kidsDF.join(teamDF, joinCondition,"right_outer")
	  allTeamsKidsDF.show()
	
	  // full outer joins
	  val fullTeamsKidsDF = kidsDF.join(teamDF, joinCondition,"full_outer")
	  fullTeamsKidsDF.show()
	
	  // semi joins
	  val allKidsWithTeamsDF = kidsDF.join(teamDF, joinCondition,"left_semi")
	  allKidsWithTeamsDF.show()
	
	  //anti join
	  val kidsWithNoTeamsDF = kidsDF.join(teamDF, joinCondition,"left_anti")
	  kidsWithNoTeamsDF.show()
	
	  //cross join
	  val productKidsWithTeams = kidsDF.crossJoin(teamDF)
	  productKidsWithTeams.show()
	
	
	  def main(args: Array[String]): Unit = {
	    Thread.sleep(10000000)
	  }
	
	}
