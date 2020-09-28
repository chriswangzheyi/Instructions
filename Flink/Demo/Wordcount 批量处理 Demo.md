# Wordcount 批量处理 Demo

## 准备数据

	I love Beijing 
	I love China
	

## 代码

### BatchWordCountDemo

	import org.apache.flink.api.common.functions.ReduceFunction
	import org.apache.flink.api.java.functions.KeySelector
	import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment
	import org.apache.flink.api.scala._
	
	object BatchWordCountDemo {
	
	  def main(args: Array[String]): Unit = {
	
	    //设置环境(Batch是ExecutionEnviroment)
	    val env= ExecutionEnvironment.getExecutionEnvironment
	
	    //定义变量
	    val inputPath = "/Users/zheyiwang/Downloads/input"
	    val outputPath = "/Users/zheyiwang/Downloads/output"
	
	
	    //读取文件
	    val text = env.readTextFile(inputPath)
	
	    val counts = text.flatMap(_.split(" "))
	      .filter(_.nonEmpty)
	      .map((_, 1))
	      .groupBy(0)
	      .sum(1)
	      .setParallelism(1)
	
	    //输出到文档
	    counts.writeAsCsv(outputPath,"\n"," ")
	
	    //启动任务
	    env.execute("batch word count")
	
	  }
	
	}

	
	
### pom.xml

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

	Beijing 1
	China 1
	I 2
	love 2





