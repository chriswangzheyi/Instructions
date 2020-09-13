# Parquest文件

## 特点

按列存储


## 转换格式

从Json到Parquest

	val empJSON = spark.read.json("/root/temp/emp.json")
	empJSON.write.mode("overwrite").parquet("/root/temp/result")
	
## 合并Schema

第一个文件

	val df1 = sc.makeRDD(1 to 5).map(i=>(i,i*2)).toDF("single","double")
	df1.write.parquet("/home/hadoop_spark/temp/test_table/key=1")

在/home/hadoop_spark/temp/test_table目录下生产了一个key=1的文件夹

	scala> df1.show
	+------+------+
	|single|double|
	+------+------+
	|     1|     2|
	|     2|     4|
	|     3|     6|
	|     4|     8|
	|     5|    10|
	+------+------+
				
第二个文件

	val df2 = sc.makeRDD(6 to 10).map(i=>(i,i*3)).toDF("single","triple")
	df2.write.parquet("/home/hadoop_spark/temp/test_table/key=2")

在/home/hadoop_spark/temp/test_table目录下生产了一个key=2的文件夹

	scala> df2.show
	+------+------+
	|single|triple|
	+------+------+
	|     6|    18|
	|     7|    21|
	|     8|    24|
	|     9|    27|
	|    10|    30|
	+------+------+
	
合并上面的文件

	val df3 = spark.read.option("mergeSchema","true").parquet("/home/hadoop_spark/temp/test_table")
	
	scala> df3.show
	+------+------+------+---+
	|single|double|triple|key|
	+------+------+------+---+
	|     8|  null|    24|  2|
	|     9|  null|    27|  2|
	|    10|  null|    30|  2|
	|     3|     6|  null|  1|
	|     4|     8|  null|  1|
	|     5|    10|  null|  1|
	|     6|  null|    18|  2|
	|     7|  null|    21|  2|
	|     1|     2|  null|  1|
	|     2|     4|  null|  1|
	+------+------+------+---+
		