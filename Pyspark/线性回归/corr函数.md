# corr函数

## 定义 

比较两个列之间的关系

## 例子

	from pyspark.sql import SparkSession
	from pyspark.sql.functions import corr
	
	if __name__ == '__main__':
	
	    spark =SparkSession.builder.appName('lin_reg').getOrCreate()
	    df = spark.read.csv('/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_4_Linear_Regression/Linear_regression_dataset.csv',inferSchema=True, header= True)
	
	    print(df.show(5,False))
	
	    ans = df.select(corr('var_1','output')).show()
	
	    print(ans)

## 显示

	+-----+-----+-----+-----+-----+------+
	|var_1|var_2|var_3|var_4|var_5|output|
	+-----+-----+-----+-----+-----+------+
	|734  |688  |81   |0.328|0.259|0.418 |
	|700  |600  |94   |0.32 |0.247|0.389 |
	|712  |705  |93   |0.311|0.247|0.417 |
	|734  |806  |69   |0.315|0.26 |0.415 |
	|613  |759  |61   |0.302|0.24 |0.378 |
	+-----+-----+-----+-----+-----+------+
	
	+-------------------+
	|corr(var_1, output)|
	+-------------------+
	| 0.9187399607627283|
	+-------------------+
	
	
输出内容表示相关性91.8%


