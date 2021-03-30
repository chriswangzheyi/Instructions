# 逻辑回归 Demo

##代码

	from pyspark.sql import SparkSession
	from pyspark.ml.feature import StringIndexer
	from pyspark.ml.feature import OneHotEncoder
	from pyspark.ml.feature import VectorAssembler
	from pyspark.ml.classification import LogisticRegression
	
	
	if __name__ == '__main__':
	
	    spark = SparkSession.builder.appName('log_reg').getOrCreate()
	
	    df = spark.read.csv('/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_5_Logistic_Regression/Log_Reg_dataset.csv',inferSchema=True,header=True)
	
	    print(df.show(5,False))
	
	    #将platfrom做vector映射
	    search_engine_indexer = StringIndexer(inputCol='Platform', outputCol='Platform_Num').fit(df)
	    df = search_engine_indexer.transform(df)
	    search_engine_encoder = OneHotEncoder(inputCol='Platform_Num',outputCol='Platform_Vector').fit(df)
	    df = search_engine_encoder.transform(df)
	
	    #将country做vector映射
	    country_indexer = StringIndexer(inputCol='Country', outputCol='Country_Num').fit(df)
	    df = country_indexer.transform(df)
	    country_encoder = OneHotEncoder(inputCol='Country_Num',outputCol='Country_Vector').fit(df)
	    df = country_encoder.transform(df)
	
	    print(df.show(5,False))
	
	    # 将所有的输入列组装成单个向量
	    df_assembler = VectorAssembler(inputCols=['Platform_Vector','Country_Vector','Age','Repeat_Visitor','Web_pages_viewed'],outputCol='features')
	    df=df_assembler.transform(df)
	
	    print(df.show(5,False))
	
	    model_df = df.select(['features','Status'])
	
	    print(model_df.show(5,False))
	
	    #划分数据集
	    training_df, test_df = model_df.randomSplit([0.75,0.25])
	
	    #构建和训练逻辑回归模型
	    log_reg = LogisticRegression(labelCol='Status').fit(training_df)
	
	    #训练结果
	    train_results = log_reg.evaluate(training_df).predictions
	
	    #测试数据集
	    results = log_reg.evaluate(test_df).predictions
	
	    # 混淆矩阵
	    tp = results[(results.Status ==1) & (results.prediction ==1)].count()
	    tn = results[(results.Status ==0) & (results.prediction ==0)].count()
	    fp = results[(results.Status ==0) & (results.prediction ==1)].count()
	    fn = results[(results.Status ==1) & (results.prediction ==0)].count()
	
	    # 准确率
	    accuracy = float( (tp+tn)/ (tp+tn+fp+fn) )
	    print("accuracy=",accuracy)
	
	    # 召回率
	    recall = float( tp/(tp+fn))
	    print("recall=",recall)
	
	    # 精度
	    prediction = float( tp/ (tp+fp))
	    print("prediction=", prediction)
	    
	    
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



	+---------+---+--------------+--------+----------------+------+------------+---------------+-----------+--------------+
	|Country  |Age|Repeat_Visitor|Platform|Web_pages_viewed|Status|Platform_Num|Platform_Vector|Country_Num|Country_Vector|
	+---------+---+--------------+--------+----------------+------+------------+---------------+-----------+--------------+
	|India    |41 |1             |Yahoo   |21              |1     |0.0         |(2,[0],[1.0])  |1.0        |(3,[1],[1.0]) |
	|Brazil   |28 |1             |Yahoo   |5               |0     |0.0         |(2,[0],[1.0])  |2.0        |(3,[2],[1.0]) |
	|Brazil   |40 |0             |Google  |3               |0     |1.0         |(2,[1],[1.0])  |2.0        |(3,[2],[1.0]) |
	|Indonesia|31 |1             |Bing    |15              |1     |2.0         |(2,[],[])      |0.0        |(3,[0],[1.0]) |
	|Malaysia |32 |0             |Google  |15              |1     |1.0         |(2,[1],[1.0])  |3.0        |(3,[],[])     |
	+---------+---+--------------+--------+----------------+------+------------+---------------+-----------+--------------+
	
	
	+---------+---+--------------+--------+----------------+------+------------+---------------+-----------+--------------+-----------------------------------+
	|Country  |Age|Repeat_Visitor|Platform|Web_pages_viewed|Status|Platform_Num|Platform_Vector|Country_Num|Country_Vector|features                           |
	+---------+---+--------------+--------+----------------+------+------------+---------------+-----------+--------------+-----------------------------------+
	|India    |41 |1             |Yahoo   |21              |1     |0.0         |(2,[0],[1.0])  |1.0        |(3,[1],[1.0]) |[1.0,0.0,0.0,1.0,0.0,41.0,1.0,21.0]|
	|Brazil   |28 |1             |Yahoo   |5               |0     |0.0         |(2,[0],[1.0])  |2.0        |(3,[2],[1.0]) |[1.0,0.0,0.0,0.0,1.0,28.0,1.0,5.0] |
	|Brazil   |40 |0             |Google  |3               |0     |1.0         |(2,[1],[1.0])  |2.0        |(3,[2],[1.0]) |(8,[1,4,5,7],[1.0,1.0,40.0,3.0])   |
	|Indonesia|31 |1             |Bing    |15              |1     |2.0         |(2,[],[])      |0.0        |(3,[0],[1.0]) |(8,[2,5,6,7],[1.0,31.0,1.0,15.0])  |
	|Malaysia |32 |0             |Google  |15              |1     |1.0         |(2,[1],[1.0])  |3.0        |(3,[],[])     |(8,[1,5,7],[1.0,32.0,15.0])        |
	+---------+---+--------------+--------+----------------+------+------------+---------------+-----------+--------------+-----------------------------------+
	
	

	+-----------------------------------+------+
	|features                           |Status|
	+-----------------------------------+------+
	|[1.0,0.0,0.0,1.0,0.0,41.0,1.0,21.0]|1     |
	|[1.0,0.0,0.0,0.0,1.0,28.0,1.0,5.0] |0     |
	|(8,[1,4,5,7],[1.0,1.0,40.0,3.0])   |0     |
	|(8,[2,5,6,7],[1.0,31.0,1.0,15.0])  |1     |
	|(8,[1,5,7],[1.0,32.0,15.0])        |1     |
	+-----------------------------------+------+


	accuracy= 0.9396447315300767
	recall= 0.9340614886731392
	prediction= 0.9443762781186094