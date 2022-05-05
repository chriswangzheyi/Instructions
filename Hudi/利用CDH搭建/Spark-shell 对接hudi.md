# Spark-shell 对接hudi

spark-shell启动,需要指定spark-avro模块，因为默认环境里没有，spark-avro模块版本还需要和spark版本对应，这里都是2.4.0。


## 授权

	hadoop fs -chmod 777 /user
	
## 解决版本不匹配问题

	vim /home/jumpuser/Hudi/hudi-spark-datasource/hudi-spark-common/src/main/scala/org/apache/hudi/DataSourceOptions.scala
	

## 保存数据血缘

	mkdir -p /var/log/spark2/lineage
	chown yarn:yarn /var/log/spark2/lineage

**重新编译**

## 启动

	spark-shell \
	--packages org.apache.spark:spark-avro_2.11:2.4.4 \
	--conf 'spark.serializer=org.apache.spark.serializer.KryoSerializer' \
	--jars /home/jumpuser/Hudi/packaging/hudi-spark-bundle/target/hudi-spark-bundle_2.11-0.9.0.jar
	
	
	
	

命令执行后：
  
          ---------------------------------------------------------------------
        |                  |            modules            ||   artifacts   |
        |       conf       | number| search|dwnlded|evicted|| number|dwnlded|
        ---------------------------------------------------------------------
        |      default     |   2   |   2   |   2   |   0   ||   2   |   2   |
        ---------------------------------------------------------------------

	:: problems summary ::
	:::: ERRORS
	        SERVER ERROR: Bad Gateway url=http://dl.bintray.com/spark-packages/maven/org/apache/apache/18/apache-18.jar
	
	        SERVER ERROR: Bad Gateway url=http://dl.bintray.com/spark-packages/maven/org/apache/spark/spark-parent_2.11/2.4.0/spark-parent_2.11-2.4.0.jar
	
	        SERVER ERROR: Bad Gateway url=http://dl.bintray.com/spark-packages/maven/org/apache/spark/spark-avro_2.11/2.4.0/spark-avro_2.11-2.4.0-javadoc.jar
	
	        SERVER ERROR: Bad Gateway url=http://dl.bintray.com/spark-packages/maven/org/sonatype/oss/oss-parent/9/oss-parent-9.jar
	
	
	:: USE VERBOSE OR DEBUG MESSAGE LEVEL FOR MORE DETAILS
	:: retrieving :: org.apache.spark#spark-submit-parent-968bf946-67a3-4b8b-a8e9-8ce6d1fbd0a2
	        confs: [default]
	        2 artifacts copied, 0 already retrieved (185kB/6ms)
	Setting default log level to "WARN".
	To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
	22/05/05 13:04:16 WARN cluster.YarnSchedulerBackend$YarnSchedulerEndpoint: Attempted to request executors before the AM has registered!
	22/05/05 13:04:16 WARN lineage.LineageWriter: Lineage directory /var/log/spark/lineage doesn't exist or is not writable. Lineage for this application will be disabled.
	Spark context Web UI available at http://cdh2:4040
	Spark context available as 'sc' (master = yarn, app id = application_1650970927739_0006).
	Spark session available as 'spark'.
	Welcome to
	      ____              __
	     / __/__  ___ _____/ /__
	    _\ \/ _ \/ _ `/ __/  '_/
	   /___/ .__/\_,_/_/ /_/\_\   version 2.4.0-cdh6.3.2
	      /_/
	         
	Using Scala version 2.11.12 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_202)
	Type in expressions to have them evaluated.
	Type :help for more information.
	
	scala> 

