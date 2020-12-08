# 提交任务到Flink

## 查看资源

![](Images/1.png)

## 上传Jar包

![](Images/2.png)

## 创建测试文档（非必须）
以word count为例，如果是容器启动的Flink，进入flink_taskmanager_1，创建测试文档

	[root@wangzheyi flink]# docker exec -it 2f023d15a7eb bash

vi /input.txt

	I love Beijing 
	I love China
	
	
## 运行程序

输入执行main函数

![](Images/3.png)


## 代码

### BatchWordCountDemo

	package com.wzy
	
	import org.apache.flink.api.scala.ExecutionEnvironment
	
	object BatchWordCountDemo {
	
	  def main(args: Array[String]): Unit = {
	
	    //设置环境(Batch是ExecutionEnviroment)
	    val env = ExecutionEnvironment.getExecutionEnvironment
	
	    //定义变量
	    //val inputPath = "/Users/zheyiwang/Downloads/input"
	    //val outputPath = "/Users/zheyiwang/Downloads/output"
	    val inputPath = "/input.txt"
	    val outputPath = "/output.txt"
	
	    //读取文件
	    val text = env.readTextFile(inputPath)
	
	    val counts = text.flatMap(_.split(" "))
	      .filter(_.nonEmpty)
	      .map((_, 1))
	      .groupBy(0)
	      .sum(1)
	      .setParallelism(1)
	
	    //输出到文档
	    counts.writeAsCsv(outputPath, "\n", " ")
	
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
	
	
	
	
	    <build>
	        <plugins>
	            <!-- 该插件用于将Scala代码编译成class文件 -->
	            <plugin>
	                <groupId>net.alchim31.maven</groupId>
	                <artifactId>scala-maven-plugin</artifactId>
	                <version>3.4.6</version>
	                <executions>
	                    <execution>
	                        <!-- 声明绑定到maven的compile阶段 -->
	                        <goals>
	                            <goal>testCompile</goal>
	                        </goals>
	                    </execution>
	                </executions>
	            </plugin>
	            <plugin>
	                <groupId>org.apache.maven.plugins</groupId>
	                <artifactId>maven-assembly-plugin</artifactId>
	                <version>3.0.0</version>
	                <configuration>
	                    <descriptorRefs>
	                        <descriptorRef>jar-with-dependencies</descriptorRef>
	                    </descriptorRefs>
	                </configuration>
	                <executions>
	                    <execution>
	                        <id>make-assembly</id>
	                        <phase>package</phase>
	                        <goals>
	                            <goal>single</goal>
	                        </goals>
	                    </execution>
	                </executions>
	            </plugin>
	        </plugins>
	    </build>
	
	</project>