# 线性回归 Demo

## 代码

	from pyspark.ml.feature import VectorAssembler
	from pyspark.sql import SparkSession
	from pyspark.ml.regression import LinearRegression
	
	if __name__ == '__main__':
	
	    spark =SparkSession.builder.appName('lin_reg').getOrCreate()
	    df = spark.read.csv('/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_4_Linear_Regression/Linear_regression_dataset.csv',inferSchema=True, header= True)
	
	    vec_assmebler= VectorAssembler( inputCols= ['var_1','var_2','var_3','var_4','var_5'], outputCol='features')
	
	    # 特征工程
	    features_df= vec_assmebler.transform(df)
	
	    # 选取建模所需feature
	    model_df = features_df.select('features','output')
	
	    print(model_df.show(5,False))
	
	    # 将样本化为训练集和测试集。
	    train_df, test_df = model_df.randomSplit([0.7,0.3])
	
	
	    lin_reg= LinearRegression(labelCol='output')
	    lr_model = lin_reg.fit(train_df)
	
	    # 回归模型的回归系数
	    print(lr_model.coefficients)
	
	    # 截距
	    print(lr_model.intercept)
	
	    #r2测定系数
	    training_predictions = lr_model.evaluate(train_df)
	    print(training_predictions.r2)
	
	    # 评估模型
	    test_results = lr_model.evaluate(test_df)
	    print(test_results.r2)
	    print(test_results.meanSquaredError)
	    
	    
## 显示


	+------------------------------+------+
	|features                      |output|
	+------------------------------+------+
	|[734.0,688.0,81.0,0.328,0.259]|0.418 |
	|[700.0,600.0,94.0,0.32,0.247] |0.389 |
	|[712.0,705.0,93.0,0.311,0.247]|0.417 |
	|[734.0,806.0,69.0,0.315,0.26] |0.415 |
	|[613.0,759.0,61.0,0.302,0.24] |0.378 |
	+------------------------------+------+
	
	# 回归模型的回归系数
	[0.0003239210625141783,5.5932645760239694e-05,0.00020067797657905847,-0.6234816099067935,0.5752769207288153]
	
	#截距
	0.1638604891309064
	
	#模型训练集测定系数
	0.8728714416082235
	
	#模型评估的测定系数
	0.8578824550941366
	
	#模型评估的mse均方误差
	0.00014881902631047847
	
均方误差：真实值与预测值差平方的期望。值越大表示效果越差，反正则效果越好。