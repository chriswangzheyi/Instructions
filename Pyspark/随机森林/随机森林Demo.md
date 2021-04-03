# 随机森林Demo

## 样本字段说明

	rate_marriage：对婚姻的评价	
	age	：年龄
	yrs_married：结婚年数	
	children	：有小孩的年数
	religious	：对宗教信仰的评分
	affairs：是否婚外情
	
## 代码

	from pyspark.sql import SparkSession
	from pyspark.ml.feature import VectorAssembler
	from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
	
	if __name__ == '__main__':
	
	    spark = SparkSession.builder.appName("random_forest").getOrCreate()
	    df = spark.read.csv('/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_6_Random_Forests/affairs.csv',
	                        inferSchema=True,header=True)
	    print(df.show(10,False))
	
	    #特征工程(组装feature)
	    df_assembler = VectorAssembler(inputCols=['rate_marriage','age','yrs_married','children','religious'],outputCol="features")
	    df = df_assembler.transform(df)
	
	    #选择模型训练所需数据
	    model_df = df.select(['features', 'affairs'])
	
	    #划分数据集
	    train_df,test_df= model_df.randomSplit([0.75,0.25])
	
	    # 构建模型
	    rf_classifier= RandomForestClassifier(labelCol='affairs',numTrees=50).fit(train_df)
	
	    # 评估模型
	    rf_predictions = rf_classifier.transform(test_df)
	
	    # 准确率
	    rf_accuracy = MulticlassClassificationEvaluator(labelCol='affairs', metricName='accuracy').evaluate(rf_predictions)
	    print("The accuracy of RF on test data is", rf_accuracy)
	
	    # 精度
	    rf_precision = MulticlassClassificationEvaluator(labelCol='affairs', metricName='weightedPrecision').evaluate(rf_predictions)
	    print("The precision rate on test data is", rf_precision)
	
	    # AUC 面积。AUC就是ROC曲线下的面积，衡量学习器优劣的一种性能指标
	    tf_auc = BinaryClassificationEvaluator(labelCol='affairs').evaluate(rf_predictions)
	    print("auc area:",tf_auc)
	
	    # 查看 特征重要性, idx越小，重要性越高
	    print(df.schema["features"].metadata["ml_attr"]["attrs"])
	
	    # 保存模型
	    rf_classifier.save("/Users/zheyiwang/Downloads/rf_model")
	
	    # 调用模型
	    rf = RandomForestClassificationModel.load("/Users/zheyiwang/Downloads/rf_model")
	    new_predictions = rf.transform(df)

## 显示


	+-------------+----+-----------+--------+---------+-------+
	|rate_marriage|age |yrs_married|children|religious|affairs|
	+-------------+----+-----------+--------+---------+-------+
	|5            |32.0|6.0        |1.0     |3        |0      |
	|4            |22.0|2.5        |0.0     |2        |0      |
	|3            |32.0|9.0        |3.0     |3        |1      |
	|3            |27.0|13.0       |3.0     |1        |1      |
	|4            |22.0|2.5        |0.0     |1        |1      |
	|4            |37.0|16.5       |4.0     |3        |1      |
	|5            |27.0|9.0        |1.0     |1        |1      |
	|4            |27.0|9.0        |0.0     |2        |1      |
	|5            |37.0|23.0       |5.5     |2        |1      |
	|5            |37.0|23.0       |5.5     |2        |1      |
	+-------------+----+-----------+--------+---------+-------+
	
	
	The accuracy of RF on test data is 0.7204502814258912
	The precision rate on test data is 0.7037537553835708
	auc area: 0.7444865259855994
	{'numeric': [{'idx': 0, 'name': 'rate_marriage'}, {'idx': 1, 'name': 'age'}, {'idx': 2, 'name': 'yrs_married'}, {'idx': 3, 'name': 'children'}, {'idx': 4, 'name': 'religious'}]}
	
	