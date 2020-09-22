# WordCount Demo

## 代码

### wordCount

	import org.apache.flink.api.scala.ExecutionEnvironment
	import org.apache.flink.streaming.api.scala.createTypeInformation
	
	object wordCount {
	
	  def main(args: Array[String]): Unit = {
	
	    val env = ExecutionEnvironment.getExecutionEnvironment
	
	    val data = env.fromElements("I love Beijing","I love China")
	
	    val count = data.flatMap(_.split(" ")).map((_,1)).groupBy(0).sum(1)
	
	    count.print()
	
	  }
	
	}


### pom

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0"
	         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	
	    <groupId>org.example</groupId>
	    <artifactId>flink_demo</artifactId>
	    <version>1.0-SNAPSHOT</version>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.apache.flink</groupId>
	            <artifactId>flink-core</artifactId>
	            <version>1.11.2</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.flink</groupId>
	            <artifactId>flink-scala_2.12</artifactId>
	            <version>1.11.2</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.flink</groupId>
	            <artifactId>flink-streaming-scala_2.12</artifactId>
	            <version>1.11.2</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.flink</groupId>
	            <artifactId>flink-clients_2.12</artifactId>
	            <version>1.11.2</version>
	        </dependency>
	
	    </dependencies>
	
	</project>

## 测试

	(I,2)
	(love,2)
	(Beijing,1)
	(China,1)