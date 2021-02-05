# Azkaban 简单Demo


## 单Job工作流flow

### 创建测试文件

vi command.job

	#command.job
	type=command
	command=echo 'hello'

### 补充：azkaban 所支持的任务种类


-  command：Linux shell 命令行任务
-  gobblin：通用数据采集工具
-  hadoopJava：运行hadoopMR 任务
-  java：原生java 任务
-  hive：支持执行hiveSQL
-  pig：pig 脚本任务
-  spark：spark 任务
-  hdfsToTeradata：把数据从hdfs 导入Teradata
-  teradataToHdfs：把数据从Teradata 导入hdfs


### 将job 资源文件打包成zip 文件

	zip command.job


### 通过azkaban的web管理平台创建project并上传job压缩包

#### 首先创建project

![](/Images/1.png)


#### 上传zip

![](/Images/2.png)

#### 执行并查看log

![](/Images/3.png)

![](/Images/4.png)


## 多Job工作流flow


### 第一个job：foo.job

	# foo.job
	type=command
	command=echo foo


### 第二个job：bar.job (依赖于foo.job)

	# bar.job
	type=command
	dependencies=foo
	command=echo bar


将两个文件打包为一个zip，并上传到azkaban中，执行

![](/Images/5.png)

可以看到有先后顺序。

打印：


	05-02-2021 11:36:20 CST foo INFO - Spawned process with id 21962
	05-02-2021 11:36:20 CST foo INFO - foo
	05-02-2021 11:36:20 CST foo INFO - Process with id 21962 completed successfully in 0 seconds.

	05-02-2021 11:36:20 CST bar INFO - Working directory: /root/azkaban_demo/azkaban-3.90.0/azkaban-solo-server/build/distributions/azkaban-solo-server-0.1.0-SNAPSHOT/executions/22
	05-02-2021 11:36:20 CST bar INFO - Spawned process with id 21966
	05-02-2021 11:36:20 CST bar INFO - bar
	05-02-2021 11:36:20 CST bar INFO - Process with id 21966 completed successfully in 0 seconds.



## HDFS

	# fs.job
	type=command
	command=/home/fantj/hadoop/bin/hadoop fs -lsr /


## Hive

test.sql

	use default;
	drop table aztest;
	create table aztest(id int,name string,age int) row format delimited fields terminated by ',' ;
	load data inpath '/aztest/hiveinput' into table aztest;
	create table azres as select * from aztest;
	insert overwrite directory '/aztest/hiveoutput' select count(1) from aztest; 


hivef.job

	# hivef.job
	type=command
	command=/home/fantj/hive/bin/hive -f 'test.sql'

## Spark

Spark：

	package com.test
	import org.apache.spark.{SparkConf, SparkContext}
	 
	object AzkabanTest extends App{
	  val conf = new SparkConf()
	  .setMaster("local[2]")
	  .setAppName("azkabanTest")
	  val sc = new SparkContext(conf)
	 
	  val data = sc.parallelize(1 to 10)
	  data.map{_ * 2}.foreach(println)
	}



vim test.job

	type=command
	command=/usr/install/spark/bin/spark-submit --class com.test.AzkabanTest test-1.0-SNAPSHOT.jar