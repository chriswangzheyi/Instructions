# Spark SQL 访问Mysql

	
## UserCaseClass.scala

	import java.util.Properties
	
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
	
	    //
	    val props = new Properties()
	    props.setProperty("user","root")
	    props.setProperty("password","wzy19910921")
	
	    //建表和插入数据
	    result.write.jdbc("jdbc:mysql://localhost:3306/test","student",props)
	
	    //如果表已经创建，新增数据，则采用append模式
	    result.write.mode("append").jdbc("jdbc:mysql://localhost:3306/test","student",props)
	
	
	
	  }
	
	}
	
	
	//定义样本类
	case class Student(stuID:Int, stuName:String, stuAge:Int){
	
	
	}
	

## pom.xml

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
	        <dependency>
	            <groupId>mysql</groupId>
	            <artifactId>mysql-connector-java</artifactId>
	            <version>8.0.21</version>
	        </dependency>
	
	    </dependencies>
	
	</project>