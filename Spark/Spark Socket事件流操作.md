# Spark Socket事件流操作


## netcat

netcat是一个基于Socket的网络通信工具。

例子：


### 服务器A

	yum -y install nc

	#绑定端口
	nc -l 9999


### 服务器B

	yum -y install nc
	
	#监听服务器A
	nc 192.168.195.130 9999


![](../Images/9.png)
![](../Images/10.png)


## Spark Streaming 充当Socket角色


![](../Images/11.png)

### SocketToPrint

	import org.apache.spark.SparkConf;
	import org.apache.spark.api.java.JavaRDD;
	import org.apache.spark.api.java.function.VoidFunction;
	import org.apache.spark.api.java.function.VoidFunction2;
	import org.apache.spark.streaming.Durations;
	import org.apache.spark.streaming.Time;
	import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
	import org.apache.spark.streaming.api.java.JavaStreamingContext;
	
	public class SocketToPrint {
	
	    public static void main(String[] args) throws InterruptedException {
	
	        SparkConf conf = new SparkConf();
	        conf.setMaster("local[2]");
	        conf.setAppName("test");
	        JavaStreamingContext jsc = new JavaStreamingContext(conf, Durations.seconds(5));
	        JavaReceiverInputDStream<String> JobLines = jsc.socketTextStream("192.168.195.130",9999);
	        JobLines.foreachRDD(new VoidFunction2<JavaRDD<String>, Time>() {
	            public void call(JavaRDD<String> stringJavaRDD, Time time) throws Exception {
	
	                long size = stringJavaRDD.count();
	                System.out.println("size="+size);
	
	                stringJavaRDD.foreach(new VoidFunction<String>() {
	                    public void call(String line) throws Exception {
	                        System.out.println(line);
	                    }
	                });
	            }
	        });
	
	        jsc.start();
	        System.out.println("-----already start---------");
	
	        jsc.awaitTermination();
	        System.out.println("------ already await -------");
	
	        jsc.close();
	        System.out.println("-----already close-------");
	    }
	
	
	}


注意：socketTextStream 方法设置被监听的服务器ip。

jobline是DStream类型，属于离散流集合。他的数据来自Socket端口的扫描获取。一个离散流的集合对象对应一个特定的时间片段，比如这里设置的5秒，收集RDD为单位进行包装。

### pom.xml


	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0"
	         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	
	    <groupId>org.example</groupId>
	    <artifactId>spark_socket_local</artifactId>
	    <version>1.0-SNAPSHOT</version>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-core_2.12</artifactId>
	            <version>2.4.3</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-streaming_2.12</artifactId>
	            <version>2.4.4</version>
	        </dependency>
	    </dependencies>
	
	</project>



### 服务器端

192.168.195.130 为例

	nc -l 9999

然后输入信息，则可以在本地接收到





## Socket 通过Spark Streaming 保存到 HDFS上


![](../Images/12.png)
	
	
### SocketToHDFS

	package com.wzy;
	
	import org.apache.hadoop.conf.Configuration;
	import org.apache.hadoop.fs.FileSystem;
	import org.apache.hadoop.fs.Path;
	import org.apache.spark.SparkConf;
	import org.apache.spark.api.java.JavaRDD;
	import org.apache.spark.api.java.function.VoidFunction;
	import org.apache.spark.streaming.Durations;
	import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
	import org.apache.spark.streaming.api.java.JavaStreamingContext;
	
	import java.io.IOException;
	import java.net.URI;
	import java.net.URISyntaxException;
	
	public class SocketToHDFS {
	
	    private static FileSystem fs;
	
	    static {
	        Configuration conf = new Configuration();
	
	        try {
	            try {
	                //此处填写hdfs master的路径
	                fs = FileSystem.get(new URI("hdfs://Master001:9000"), conf, "root");
	            } catch (IOException e) {
	                e.printStackTrace();
	            } catch (InterruptedException e) {
	                e.printStackTrace();
	            } catch (URISyntaxException e) {
	                e.printStackTrace();
	            }
	        } finally {
	
	        }
	    }
	
	
	    public static void main(String[] args) throws IOException, InterruptedException {
	
	        Path output = new Path("/spark/streaming/output");
	        if (fs.exists(output)){
	            fs.delete(output,true);
	        }
	
	        SparkConf conf = new SparkConf();
	        conf.setAppName("test");
	        JavaStreamingContext jsc = new JavaStreamingContext(conf, Durations.seconds(5));
	        JavaReceiverInputDStream<String> Jobslines = jsc.socketTextStream("192.168.195.130",9999);
	
	        Jobslines.foreachRDD(new VoidFunction<JavaRDD<String>>() {
	            public void call(JavaRDD<String> stringJavaRDD) throws Exception {
	                long size = stringJavaRDD.count();
	                System.out.println("--------foreachRDD-call-collection-size:"+size);
	
	                if (size !=0){
	                    stringJavaRDD.saveAsTextFile("/spark/streaming/output");
	                }
	
	            }
	        });
	
	        jsc.start();
	        jsc.awaitTermination();
	        jsc.close();
	
	    }
	
	}


### pom

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0"
	         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	
	    <groupId>org.example</groupId>
	    <artifactId>spark_socket_hdfs</artifactId>
	    <version>1.0-SNAPSHOT</version>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-core_2.12</artifactId>
	            <version>2.4.3</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-streaming_2.12</artifactId>
	            <version>2.4.4</version>
	        </dependency>
	    </dependencies>
	
	</project>


### 部署步骤

打包项目，并上传到服务器root目录下

	转到spark目录
	cd /root/spark-2.4.4-bin-hadoop2.7/bin
	
	#使用spark submit提交任务
	./spark-submit --class com.wzy.SocketToHDFS --master spark://Master001:7077 --executor-memory 512M --total-executor-cores 2 /root/spark_socket_hdfs-1.0-SNAPSHOT.jar 

	#如果Master服务器的spark没有启动
	cd /root/spark-2.4.4-bin-hadoop2.7/sbin
	. start-master.sh

	#如果Slave服务器的Spark没有启动
	./root/spark-2.4.4-bin-hadoop2.7/sbin/start-slave.sh {spark master:7077}

	cd /root/spark-2.4.4-bin-hadoop2.7/sbin
	./start-slave.sh 192.168.195.128:7077 


在服务器端（192.168.195.130）输入命令：
	
	nc -l 9999

	#然后输入测试文本
	wangzheyi is tesing

在hadopp端，查看

	#查看是否生成输出文件
	hdfs dfs -ls /spark/streaming/output
	
	#查看输出
	hdfs dfs -cat /spark/streaming/output/part-00001

从结果可以看到，HDFS保存的文件中，会不停的被RDD覆盖。如果想要不被覆盖，需要在Taskzhong shiyong FileSystem的I/O流来覆盖。



## Spark Streaming 通过 I/O 

为了解决上一个项目，只有最后的消息被保存的问题，需通过I/O来保存文件。


![](../Images/14.png)


### SocketToHDFS

	
	package com.wzy;
	
	import org.apache.hadoop.conf.Configuration;
	import org.apache.hadoop.fs.FSDataOutputStream;
	import org.apache.hadoop.fs.FileSystem;
	import org.apache.hadoop.fs.Path;
	import org.apache.spark.SparkConf;
	import org.apache.spark.api.java.JavaRDD;
	import org.apache.spark.api.java.function.VoidFunction;
	import org.apache.spark.streaming.Durations;
	import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
	import org.apache.spark.streaming.api.java.JavaStreamingContext;
	
	import java.io.IOException;
	import java.net.URI;
	import java.net.URISyntaxException;
	import java.util.Iterator;
	
	public class SocketToHDFS {
	
	    private static FileSystem fs;
	
	    private static int count;
	    private static final String  outPath="/spark/streaming/output";
	
	    static {
	        Configuration conf = new Configuration();
	
	        try {
	            try {
	                //此处填写hdfs master的路径
	                fs = FileSystem.get(new URI("hdfs://Master001:9000"), conf, "root");
	            } catch (IOException e) {
	                e.printStackTrace();
	            } catch (InterruptedException e) {
	                e.printStackTrace();
	            } catch (URISyntaxException e) {
	                e.printStackTrace();
	            }
	        } finally {
	
	        }
	    }
	
	    private static void saveLine(Iterator<String> its) throws IOException{
	        Path outFile = new Path(outPath + "/Part-"+count++);
	        FSDataOutputStream dos = fs.create(outFile, true);
	
	        try{
	            while (its.hasNext()){
	                String line = its.next();
	                dos.writeUTF(line+"\n");
	                dos.flush();
	            }
	        }finally {
	            dos.close();
	        }
	    }
	
	    public static void main(String[] args) throws IOException, InterruptedException {
	
	        Path output = new Path(outPath);
	        if (fs.exists(output)){
	            fs.delete(output,true);
	        }
	
	        boolean flag = fs.mkdirs(output);
	        if (!flag) return;
	
	        SparkConf conf = new SparkConf();
	        conf.setAppName("test");
	        JavaStreamingContext jsc = new JavaStreamingContext(conf, Durations.seconds(5));
	        JavaReceiverInputDStream<String> Jobslines = jsc.socketTextStream("192.168.195.130",9999);
	
	        Jobslines.foreachRDD(new VoidFunction<JavaRDD<String>>() {
	            public void call(JavaRDD<String> stringJavaRDD) throws Exception {
	                long size = stringJavaRDD.count();
	                System.out.println("--------foreachRDD-call-collection-size:"+size);
	
	                if(0==size){
	                    return;
	                }
	
	                if (size !=0){
	                    stringJavaRDD.foreachPartition(new VoidFunction<Iterator<String>>() {
	                        public void call(Iterator<String> its) throws Exception {
	                            saveLine(its);
	                        }
	                    });
	                }
	
	            }
	        });
	
	        jsc.start();
	        jsc.awaitTermination();
	        jsc.close();
	
	    }
	
	}


### pom.xml


	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0"
	         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	
	    <groupId>org.example</groupId>
	    <artifactId>spark_socket_hdfs</artifactId>
	    <version>1.0-SNAPSHOT</version>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-core_2.12</artifactId>
	            <version>2.4.3</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.spark</groupId>
	            <artifactId>spark-streaming_2.12</artifactId>
	            <version>2.4.4</version>
	        </dependency>
	    </dependencies>
	
	</project>



### 部署步骤

	转到spark目录
	cd /root/spark-2.4.4-bin-hadoop2.7/bin
	
	#使用spark submit提交任务
	./spark-submit --class com.wzy.SocketToHDFS --master spark://Master001:7077 --executor-memory 512M --total-executor-cores 2 /root/spark_socket_hdfs-1.0-SNAPSHOT.jar 

	#如果Master服务器的spark没有启动
	cd /root/spark-2.4.4-bin-hadoop2.7/sbin
	. start-master.sh

	#如果Slave服务器的Spark没有启动
	./root/spark-2.4.4-bin-hadoop2.7/sbin/start-slave.sh {spark master:7077}

	cd /root/spark-2.4.4-bin-hadoop2.7/sbin
	./start-slave.sh 192.168.195.128:7077 

在服务器端（192.168.195.130）输入命令：
	
	nc -l 9999

	#然后输入测试文本
	wangzheyi is tesing

在hadopp端，查看

	#查看是否生成输出文件
	hdfs dfs -ls /spark/streaming/output
	
	#查看输出
	hdfs dfs -cat /spark/streaming/output/Part-1

![](../Images/13.png)


**可以看到，一个文件保存netcat一行输入的内容**