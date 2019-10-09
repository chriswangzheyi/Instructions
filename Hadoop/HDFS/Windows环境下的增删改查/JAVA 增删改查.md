#Java 增删改查


# Java程序

![](../Images/3.png)


**ReadWriteHDFSExample:**


	package com.wzy.hdfs;
	
	import org.apache.commons.io.IOUtils;
	import org.apache.hadoop.conf.Configuration;
	import org.apache.hadoop.fs.FSDataInputStream;
	import org.apache.hadoop.fs.FSDataOutputStream;
	import org.apache.hadoop.fs.FileSystem;
	import org.apache.hadoop.fs.Path;
	
	import java.io.*;
	import java.nio.charset.StandardCharsets;
	
	public class ReadWriteHDFSExample {
	
	    public static void main(String[] args) throws IOException {
	
	//        在本地运行需要设置Hadoop Home的路径
	        System.setProperty("hadoop.home.dir", "D:\\hadoop-3.2.1");
	
	        ReadWriteHDFSExample.checkExists();
	//        ReadWriteHDFSExample.createDirectory();
	//        ReadWriteHDFSExample.checkExists();
	//        ReadWriteHDFSExample.writeFileToHDFS();
	//        ReadWriteHDFSExample.appendToHDFSFile();
	//        ReadWriteHDFSExample.readFileFromHDFS();
	    }
	
	    public static void readFileFromHDFS() throws IOException {
	        Configuration configuration = new Configuration();
	        configuration.set("fs.defaultFS", "hdfs://47.112.142.231:9000");
	        FileSystem fileSystem = FileSystem.get(configuration);
	        //Create a path
	        String fileName = "read_write_hdfs_example.txt";
	        Path hdfsReadPath = new Path("/user/javadeveloperzone/javareadwriteexample/" + fileName);
	        //Init input stream
	        FSDataInputStream inputStream = fileSystem.open(hdfsReadPath);
	        //Classical input stream usage
	        String out= IOUtils.toString(inputStream, "UTF-8");
	        System.out.println(out);
	
	        /*BufferedReader bufferedReader = new BufferedReader(
	                new InputStreamReader(inputStream, StandardCharsets.UTF_8));
	
	        String line = null;
	        while ((line=bufferedReader.readLine())!=null){
	            System.out.println(line);
	        }*/
	
	        inputStream.close();
	        fileSystem.close();
	    }
	
	    public static void writeFileToHDFS() throws IOException {
	        Configuration configuration = new Configuration();
	        configuration.set("fs.defaultFS", "hdfs://47.112.142.231:9000");
	        FileSystem fileSystem = FileSystem.get(configuration);
	        //Create a path
	        String fileName = "read_write_hdfs_example.txt";
	        Path hdfsWritePath = new Path("/user/javadeveloperzone/javareadwriteexample/" + fileName);
	        FSDataOutputStream fsDataOutputStream = fileSystem.create(hdfsWritePath,true);
	
	        BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(fsDataOutputStream,StandardCharsets.UTF_8));
	        bufferedWriter.write("Java API to write data in HDFS");
	        bufferedWriter.newLine();
	        bufferedWriter.close();
	        fileSystem.close();
	    }
	
	    public static void appendToHDFSFile() throws IOException {
	        Configuration configuration = new Configuration();
	        configuration.set("fs.defaultFS", "hdfs://47.112.142.231:9000");
	        FileSystem fileSystem = FileSystem.get(configuration);
	        //Create a path
	        String fileName = "read_write_hdfs_example.txt";
	        Path hdfsWritePath = new Path("/user/javadeveloperzone/javareadwriteexample/" + fileName);
	        FSDataOutputStream fsDataOutputStream = fileSystem.append(hdfsWritePath);
	
	        BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(fsDataOutputStream,StandardCharsets.UTF_8));
	        bufferedWriter.write("Java API to append data in HDFS file");
	        bufferedWriter.newLine();
	        bufferedWriter.close();
	        fileSystem.close();
	    }
	
	    public static void createDirectory() throws IOException {
	        Configuration configuration = new Configuration();
	        configuration.set("fs.defaultFS", "hdfs://47.112.142.231:9000");
	        FileSystem fileSystem = FileSystem.get(configuration);
	        String directoryName = "javadeveloperzone/javareadwriteexample";
	        Path path = new Path(directoryName);
	        fileSystem.mkdirs(path);
	    }
	
	    public static void checkExists() throws IOException {
	        Configuration configuration = new Configuration();
	        configuration.set("fs.defaultFS", "hdfs://47.112.142.231:9000");
	        FileSystem fileSystem = FileSystem.get(configuration);
	        String directoryName = "javadeveloperzone/javareadwriteexample";
	        Path path = new Path(directoryName);
	        if(fileSystem.exists(path)){
	            System.out.println("File/Folder Exists : "+path.getName());
	        }else{
	            System.out.println("File/Folder does not Exists : "+path.getName());
	        }
	    }
	}



**pom.xml**


	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.1.9.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>hdfs</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>hdfs</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter</artifactId>
	        </dependency>
	
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-test</artifactId>
	            <scope>test</scope>
	        </dependency>
	
	        <!-- https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-mapreduce-client-core -->
	        <dependency>
	            <groupId>org.apache.hadoop</groupId>
	            <artifactId>hadoop-mapreduce-client-core</artifactId>
	            <version>3.1.1</version>
	        </dependency>
	
	        <!-- https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-common -->
	        <dependency>
	            <groupId>org.apache.hadoop</groupId>
	            <artifactId>hadoop-common</artifactId>
	            <version>3.1.1</version>
	        </dependency>
	
	        <!--
	        hadoop-mapreduce-client-jobclient dependency for local debug
	        -->
	        <dependency>
	            <groupId>org.apache.hadoop</groupId>
	            <artifactId>hadoop-mapreduce-client-jobclient</artifactId>
	            <version>3.1.1</version>
	        </dependency>
	
	    </dependencies>
	
	    <build>
	        <plugins>
	            <plugin>
	                <groupId>org.springframework.boot</groupId>
	                <artifactId>spring-boot-maven-plugin</artifactId>
	            </plugin>
	        </plugins>
	    </build>
	
	</project>




##验证

使用createDirectory方法后，访问{hadoop所在主机}:50070


例如：

	http://47.112.142.231:50070



![](../Images/4.png)


---


![](../Images/5.png)
