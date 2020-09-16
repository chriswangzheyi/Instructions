# Spark Streaming Demo

## 代码

### MyNetworkWordCount


	import org.apache.spark.SparkConf
	import org.apache.spark.storage.StorageLevel
	import org.apache.spark.streaming.{Seconds, StreamingContext}
	
	object MyNetworkWordCount {
	
	  def main(args: Array[String]): Unit = {
	
	    //创建spacrkconf对象。 local表示只有一个线程，local[2]表示cpu有两个核，作用分别是处理数据、监听数据
	    val sparkConf = new SparkConf().setAppName("Network word count Demo").setMaster("local[2]")
	
	    //创建一个StreamingContext,每隔3秒采集一次数据
	    val ssc = new StreamingContext(sparkConf,Seconds(3))
	
	    //创建Dstream，看成是一个输入流
	    val line = ssc.socketTextStream("192.168.2.101",1234,StorageLevel.MEMORY_AND_DISK)
	
	    //执行word count
	    val words = line.flatMap(_.split(" "))
	    val wordCount = words.map((_,1)).reduceByKey(_+_)
	
	    //输出结果
	    wordCount.print()
	
	    //启动StreamingContext
	    ssc.start()
	
	    //等待计算完成
	    ssc.awaitTermination()
	
	  }
	
	}

### pom.xml

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
	            <artifactId>spark-streaming_2.12</artifactId>
	            <version>3.0.1</version>
	        </dependency>
	
	    </dependencies>
	
	</project>
	
## 测试

执行：

	nc -l -p 1234
	
输入内容，可以在程序中看到处理结果