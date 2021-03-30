# OneHotEncoder 函数

## 定义

将数据变为独热码

## 例子

	from pyspark.sql import SparkSession
	from pyspark.ml.feature import StringIndexer
	from pyspark.ml.feature import OneHotEncoder
	
	if __name__ == '__main__':
	
	    spark = SparkSession.builder.appName('log_reg').getOrCreate()
	
	    df = spark.read.csv('/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_5_Logistic_Regression/Log_Reg_dataset.csv',inferSchema=True,header=True)
	
	    print(df.show(5,False))
	
	    search_engine_indexer = StringIndexer(inputCol='Platform', outputCol='Platform_Num').fit(df)
	
	    df = search_engine_indexer.transform(df)
	
	    search_engine_encoder = OneHotEncoder(inputCol='Platform_Num',outputCol='Platform_Vector').fit(df)
	
	    df = search_engine_encoder.transform(df)
	
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


	+---------+---+--------------+--------+----------------+------+------------+---------------+
	|Country  |Age|Repeat_Visitor|Platform|Web_pages_viewed|Status|Platform_Num|Platform_Vector|
	+---------+---+--------------+--------+----------------+------+------------+---------------+
	|India    |41 |1             |Yahoo   |21              |1     |0.0         |(2,[0],[1.0])  |
	|Brazil   |28 |1             |Yahoo   |5               |0     |0.0         |(2,[0],[1.0])  |
	|Brazil   |40 |0             |Google  |3               |0     |1.0         |(2,[1],[1.0])  |
	|Indonesia|31 |1             |Bing    |15              |1     |2.0         |(2,[],[])      |
	|Malaysia |32 |0             |Google  |15              |1     |1.0         |(2,[1],[1.0])  |
	+---------+---+--------------+--------+----------------+------+------------+---------------+

## 解读

	(2,[0],[1.0]) represents a vector of length 2 , with 1 value: size of vector 2
	value contained in vector 1
	position of 1 value in vector 0th place

这类表示形式可以节约计算空间，因为计算耗时较短。向量的长度等于元素总元素减一。

常见的独热码表现形式为：

	            google       yahoo     bing
	google        1           0         0
	yahoo         0           1         0    
	bing          0           0         1      

以最优的方式：


	            google     yahoo    
	google        1           0        
	yahoo         0           1          
	bing          0           0              
