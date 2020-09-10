# Spark的检查机制

checkpoint的意思就是建立检查点,类似于快照,例如在spark计算里面 计算流程DAG特别长,服务器需要将整个DAG计算完成得出结果,但是如果在这很长的计算流程中突然中间算出的数据丢失了,spark又会根据RDD的依赖关系从头到尾计算一遍,这样子就很费性能,当然我们可以将中间的计算结果通过cache或者persist放到内存或者磁盘中,但是这样也不能保证数据完全不会丢失,存储的这个内存出问题了或者磁盘坏了,也会导致spark从头再根据RDD计算一遍,所以就有了checkpoint,其中checkpoint的作用就是将DAG中比较重要的中间数据做一个检查点将结果存储到一个高可用的地方(通常这个地方就是HDFS里面)


## 本地模式的检查点

 要求spark-shell是本地模式
 
	cd /home/hadoop_spark
	mkdir checkpoint
	cd /home/hadoop_spark/spark-3.0.0-bin-hadoop3.2/bin
	./spark-shell

在scala中：

####正常情况

	scala> sc.setCheckpointDir("/home/hadoop_spark/checkpoint")  
	
	scala> val rdd1 = sc.textFile("hdfs://192.168.2.101:9000/data/data.txt") 
	rdd1: org.apache.spark.rdd.RDD[String] = hdfs://192.168.2.101:9000/data/data.txt MapPartitionsRDD[3] at textFile at <console>:24
	
	scala> rdd1.checkpoint 
	
	scala> rdd1.count 
	res5: Long = 6                                                                  

/home/hadoop_spark/checkpoint 文件夹下生成文件

	-rw-r--r--. 1 hadoop_spark hadoop_spark 46 Sep  5 16:26 part-00000
	-rw-r--r--. 1 hadoop_spark hadoop_spark 35 Sep  5 16:26 part-00001

文件内记录了计算过程中的一些内容。

#### 错误情况

模拟错误：文件找不到

	scala> sc.setCheckpointDir("/home/hadoop_spark/checkpoint")  
	
	scala> val rdd1 = sc.textFile("hdfs://localhost:9000/data/data.txt")  
	rdd1: org.apache.spark.rdd.RDD[String] = hdfs://localhost:9000/data/data.txt MapPartitionsRDD[1] at textFile at <console>:24
	
	scala> rdd1.checkpoint 
	
	scala> rdd1.count 
	java.net.ConnectException: Call From wangzheyi/192.168.2.101 to localhost:9000 failed on connection exception: java.net.ConnectException: Connection refused; For more details see:  http://wiki.apache.org/hadoop/ConnectionRefused
	  at sun.reflect.NativeConstructorAccessorImpl.newInstance0(Native Method)
	  （以下略）


/home/hadoop_spark/checkpoint 文件夹下不生成文件



## HDFS的检查点

要求spark-shell是集群模式
 
 
 
 