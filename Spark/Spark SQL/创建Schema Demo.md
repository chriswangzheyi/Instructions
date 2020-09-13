# 创建Schema Demo

##使用Spark Session

### SpecifySchema

	import org.apache.spark.sql.{Row, SparkSession}
	import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
	
	object SpecifySchema {
	
	  def main(args: Array[String]): Unit = {
	
	    //创建一个Spark Session
	    val spark = SparkSession.builder().master("local").appName("specifySchema").getOrCreate();
	
	    //Spark Session包含SparkContect对象，读取数据
	    val data = spark.sparkContext.textFile("/Users/zheyiwang/Documents/GitHub/Instructions/Spark/Spark\\ SQL/student.txt")
	    .map(_.split(" "))
	
	    //创建Schema结构
	    val schema = StructType(
	      List(
	        StructField ("id", IntegerType, true),
	        StructField ("name", StringType, true),
	        StructField ("age", IntegerType, true)
	      )
	    )
	
	    //将数据映射成Row
	    val rowRDD = data.map( p => Row(p(0).toInt, p(1).trim, p(2).toInt))
	
	    //关联Schema
	    val studentDF = spark.createDataFrame(rowRDD, schema)
	
	    //生成一个表
	    studentDF.createOrReplaceTempView("student")
	
	    //执行sql
	    val result = spark.sql("select * from student")
	
	    //显示
	    result.show()
	
	    spark.stop()
	  }
	
	}


### pom

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0"
	         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	
	    <groupId>org.example</groupId>
	    <artifactId>webVisitCount</artifactId>
	    <version>1.0-SNAPSHOT</version>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-core_2.12</artifactId>
	            <version>3.0.1</version>
	        </dependency>
	        <dependency>
	            <groupId>org.scala-lang</groupId>
	            <artifactId>scala-library</artifactId>
	            <version>2.12.12</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-sql_2.12</artifactId>
	            <version>3.0.1</version>
	        </dependency>
	
	    </dependencies>
	
	</project>
	
	
### 测试结果

	+---+----+---+
	| id|name|age|
	+---+----+---+
	|  1| Tom| 23|
	|  2|Mary| 24|
	+---+----+---+
	
	
## 使用case class

只需要替换SpecifySchema为UserCaseClass.scala


### UserCaseClass.scala

	import org.apache.spark.sql.SparkSession
	
	object UserCaseClass {
	
	  def main(args: Array[String]): Unit = {
	
	    //创建一个Spark Session
	    val sparkSession = SparkSession.builder().master("local").appName("specifySchema").getOrCreate()
	
	    //Spark Session包含SparkContect对象，读取数据
	    val lineRDD = sparkSession.sparkContext.textFile("/Users/zheyiwang/Documents/GitHub/Instructions/Spark/Spark\\ SQL/student.txt")
	      .map(_.split(" "))
	
	    val studentRDD =  lineRDD.map(x => Student( x(0).toInt, x(1),x(2).toInt ))
	
	    //生成DataFrame,需要导入隐式转换
	    import sparkSession.sqlContext.implicits._
	    val studentDF = studentRDD.toDF()
	
	    //生成表
	    studentDF.createOrReplaceTempView("mystudent")
	
	    //查询
	    val result = sparkSession.sql("select * from mystudent")
	
	    //显示
	    result.show()
	
	    sparkSession.stop()
	  }
	}
	
	
	//定义样本类
	case class Student(stuID:Int, stuName:String, stuAge:Int){
	
	
	}