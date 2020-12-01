# wordcount 滑动窗口Demo

## 代码

### SocketWindowWordCount

	import org.apache.flink.streaming.api.scala.StreamExecutionEnvironment
	import org.apache.flink.api.scala._
	import org.apache.flink.streaming.api.windowing.time.Time
	
	object SocketWindowWordCount {
	
	  def main(args: Array[String]): Unit = {
	
	  //设置环境
	  val env= StreamExecutionEnvironment.getExecutionEnvironment
	
	  //设置数据源（文本流）
	  val text = env.socketTextStream("localhost",1234)
	
	    //每两秒执行一次，搜集前两秒的信息内容
	    val windowCounts = text.flatMap( line => line.split(" "))
	      .map( (_,1))
	      .keyBy(0)
	      .timeWindow(Time.seconds(2),Time.seconds(1))
	      .sum(1)
	
	    //设置并行度：设置Flink启动多少个线程去执行程序
	    windowCounts.print.setParallelism(1)
	
	    env.execute("socket window count")
	
	  }
	
	}

流式处理没有groupby，用keyBy替代。

如果不设置timeWindow，则任务会统计生命周期内的全部word

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


## 测试结果


### 客户端

（2秒内每隔一秒连续输入）

	I love Beijing
	I love Beijing

### idea显示结果

	(I,1)
	(Beijing,1)
	(love,1)
	(love,2)
	(Beijing,2)
	(I,2)
	(love,1)
	(I,1)
	(Beijing,1)
	
	
	
可以看到，统计分为1-2-1三次层次。第一个1是第一个I love beijing的统计，第二个2是两个I love beijing的统计，第三个1是第二个I love beijing的统计