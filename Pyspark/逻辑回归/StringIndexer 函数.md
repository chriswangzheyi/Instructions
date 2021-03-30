# StringIndexer 函数

## 定义

将列标记为数字。因为机器学习无法识别String，需要转换为数字。

## demo

	from pyspark.sql import SparkSession
	from pyspark.ml.feature import StringIndexer
	
	if __name__ == '__main__':
	
	    spark = SparkSession.builder.appName('log_reg').getOrCreate()
	
	    df = spark.read.csv('/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_5_Logistic_Regression/Log_Reg_dataset.csv',inferSchema=True,header=True)
	
	    print(df.show(5,False))
	
	    search_engine_indexer = StringIndexer(inputCol='Platform', outputCol='Platform_Num').fit(df)
	
	    df = search_engine_indexer.transform(df)
	
	    print(df.show(5,False))



## 显示

	+---------+---+--------------+--------+----------------+------+
	|Country  |Age|Repeat_Visitor|Platform|Web_pages_viewed|Status|
	+---------+---+--------------+--------+----------------+------+
	|India    |41 |1             |Yahoo   |21              |1     |
	|Brazil   |28 |1             |Yahoo   |5               |0     |
	|Brazil   |40 |0             |Google  |3               |0     |
	|Indonesia|31 |1             |Bing    |15              |1     |
	|Malaysia |32 |0             |Google  |15              |1     |
	+---------+---+--------------+--------+----------------+------+
	

	+---------+---+--------------+--------+----------------+------+------------+
	|Country  |Age|Repeat_Visitor|Platform|Web_pages_viewed|Status|Platform_Num|
	+---------+---+--------------+--------+----------------+------+------------+
	|India    |41 |1             |Yahoo   |21              |1     |0.0         |
	|Brazil   |28 |1             |Yahoo   |5               |0     |0.0         |
	|Brazil   |40 |0             |Google  |3               |0     |1.0         |
	|Indonesia|31 |1             |Bing    |15              |1     |2.0         |
	|Malaysia |32 |0             |Google  |15              |1     |1.0         |
	+---------+---+--------------+--------+----------------+------+------------+