# Spark SQL

## 修改配置文件

vi /root/spark-2.4.4-bin-hadoop2.7/conf/hive-site.xml

	#插入
	<configuration>
	<property>
		<name>hive.metastore.uris</name>
		<value>thrift://Master001:9083</value>
	</property>
	</configuration>


## 前置条件

- 启动mysql
- 启动hadoop集群
- 启动spark

启动Hive metastore服务(启动后支持远程访问)

	hive --service metastore -p 9083


# Spark 操作Hive

## 启动

	cd /root/spark-2.4.4-bin-hadoop2.7/bin
	./spark-shell --master spark://Master001:7077

## 测试

进入spark后：

	#创建HiveContext对象，参数是SparkContext类型的对象
	val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)

	#切换数据库到Hive， 注意hive是hive的一个database
	hiveContext.sql("use test") 

	# 显示Hive中所有的表
	hiveContext.sql("show tables").collect.foreach(println)

	# 执行数据查询并显示结果
	hiveContext.sql("select count(*) from student").collect.foreach(println)
	