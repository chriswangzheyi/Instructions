# VectorAssembler 函数

## 定义

是基础特征处理类，将多个数值列按顺序汇总成一个向量列。

## 例子

	from pyspark.ml.feature import VectorAssembler
	from pyspark.sql import SparkSession
	
	if __name__ == '__main__':
	
	    spark =SparkSession.builder.appName('lin_reg').getOrCreate()
	    df = spark.read.csv('/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_4_Linear_Regression/Linear_regression_dataset.csv',inferSchema=True, header= True)
	
	    print(df.show(5,False))
	
	    vec_assmebler= VectorAssembler( inputCols= ['var_1','var_2','var_3','var_4','var_5'], outputCol='features')
	
	    features_df= vec_assmebler.transform(df)
	
	    print(features_df.select('features').show(5,False) )
	    
	    
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
	
	
	+------------------------------+
	|features                      |
	+------------------------------+
	|[734.0,688.0,81.0,0.328,0.259]|
	|[700.0,600.0,94.0,0.32,0.247] |
	|[712.0,705.0,93.0,0.311,0.247]|
	|[734.0,806.0,69.0,0.315,0.26] |
	|[613.0,759.0,61.0,0.302,0.24] |
	+------------------------------+