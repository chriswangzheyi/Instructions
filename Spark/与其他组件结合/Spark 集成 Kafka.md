# Spark 集成 Kafka

## 代码

### KafkaToSpark

	package com.wzy
	
	import org.apache.spark.streaming.kafka010.{ConsumerStrategies, KafkaUtils, LocationStrategies}
	import org.apache.spark.{SparkConf}
	import org.apache.spark.streaming.{Seconds, StreamingContext}
	
	import scala.collection.mutable
	
	object KafkaToSpark {
	
	  def main(args: Array[String]): Unit = {
	
	    //创建spacrkconf对象
	    val conf = new SparkConf().setAppName("KafkaToSpark Demo").setMaster("local")
	
	    //创建sparkcontext
	    val ssc = new StreamingContext(conf, Seconds(5))
	
	    //配置kafka参数
	    val topicsSet = Array("kafka_spark")
	    val kafkaParams = mutable.HashMap[String, String]()
	
	    kafkaParams.put("bootstrap.servers", "10.161.50.178:9092")
	    kafkaParams.put("group.id", "group1")
	    kafkaParams.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
	    kafkaParams.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
	
	
	    val messages = KafkaUtils.createDirectStream[String, String](
	      ssc,
	      LocationStrategies.PreferConsistent,
	      ConsumerStrategies.Subscribe[String, String](topicsSet, kafkaParams)
	    )
	
	    // Get the lines, split them into words, count the words and print
	    val lines = messages.map(_.value)
	    val words = lines.flatMap(_.split(" "))
	    val wordCounts = words.map(x => (x, 1L)).reduceByKey(_ + _)
	    wordCounts.print()
	
	    // Start the computation
	    ssc.start()
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
	    <artifactId>spark_kafka</artifactId>
	    <version>1.0-SNAPSHOT</version>
	
	    <properties>
	        <maven.compiler.source>1.7</maven.compiler.source>
	        <maven.compiler.target>1.7</maven.compiler.target>
	        <encoding>UTF-8</encoding>
	
	        <!-- 这里对jar包版本做集中管理 -->
	        <scala.version>2.11.12</scala.version>
	        <spark.version>2.4.4</spark.version>
	
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <!-- scala语言核心包 -->
	            <groupId>org.scala-lang</groupId>
	            <artifactId>scala-library</artifactId>
	            <version>${scala.version}</version>
	        </dependency>
	        <dependency>
	            <!-- spark核心包 -->
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-core_2.12</artifactId>
	            <version>${spark.version}</version>
	        </dependency>
	
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-streaming_2.12</artifactId>
	            <version>2.4.2</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-streaming-kafka-0-10_2.12</artifactId>
	            <version>2.4.3</version>
	        </dependency>
	        
	    </dependencies>
	
	</project>


## 启动Kafka

	kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 3 --topic kafka_spark
	//查看创建的topic，有记录说明创建成功
	kafka-topics.sh --list --zookeeper localhost:2181

### 启动生成者，向topic中生产数据
	
	./kafka-console-producer.sh --broker-list localhost:9092 --topic kafka_spark