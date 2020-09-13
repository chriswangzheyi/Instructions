# Load 和 Save 函数

load和save函数： 默认都是Parquet文件

## Load

使用load函数加载数据，自动生成表（DataFrame）

	scala> val usersDF = spark.read.load("/home/hadoop_spark/spark-3.0.0-bin-hadoop3.2/examples/src/main/resources/users.parquet")
	usersDF: org.apache.spark.sql.DataFrame = [name: string, favorite_color: string ... 1 more field]
	
	scala> usersDF.show
	+------+--------------+----------------+                                        
	|  name|favorite_color|favorite_numbers|
	+------+--------------+----------------+
	|Alyssa|          null|  [3, 9, 15, 20]|
	|   Ben|           red|              []|
	+------+--------------+----------------+
	

### 对比：

如果是Json等其他格式：

	scala> spark.read.json("/home/hadoop_spark/spark-3.0.0-bin-hadoop3.2/examples/src/main/resources/people.json")
	res2: org.apache.spark.sql.DataFrame = [age: bigint, name: string]

不使用Load函数，直接调用特定的格式即可。

## Save

将查询结果保存为Parquet文件：

	scala> usersDF.select($"name",$"favorite_color").write.save("/home/hadoop_spark/temp/result")