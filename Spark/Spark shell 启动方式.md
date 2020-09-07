# Spark shell 启动方式

## 路径

spark下的bin目录


##单机启动

./spark-shell

显示：

	Spark context available as 'sc' (master = local[*], app id = local-1599198789548).

##集群启动

./spark-shell --master spark://192.168.2.101:7077

显示：

	Spark context available as 'sc' (master = 92.168.2.101, app id = app-1599198789548).
	
	
	
## Demo

### 准备数据

	hdfs dfs -cat /data/data.txt
	
数据如下：

	hadoop hive
	hive hadoop
	hbase sqoop
	hbase sqoop
	hadoop hive
	
###  scala shell 代码

	sc.textFile("hdfs://192.168.2.101:9000/data/data.txt").flatMap(_.split(" ")).map((_,1)).reduceByKey(_+_).saveAsTextFile("hdfs://192.168.2.101:9000/output/spark/wc1") 

解读：

	sc：代表SparkContext对象，该对象是提交Spark任务的入口
					
	sc.textFile("hdfs://192.168.2.101:9000/data/data.txt") :    从HDFS中读取数据，并生成一个RDD          
	
	.flatMap(_.split(" "))  对数据压平，并分词 ---->  (I) (hadoop) (hive)
					  
	.map((_,1)) : 对每个单词，生产一个元组对 -----> (hadoop,1) (hive,1) (hbase,1)
	
	.reduceByKey(_+_) : 按照key进行reduce，将value进行累加
					  
	.saveAsTextFile("hdfs://hadoop111:9000/output/spark/wc1"):    将结果存入HDFS


### 查看结果

查看是否生成了对应文件

	hdfs dfs -ls /output/spark/wc1/

查看统计结果

	hdfs dfs -cat /output/spark/wc1/part-00000
	hdfs dfs -cat /output/spark/wc1/part-00001
				  