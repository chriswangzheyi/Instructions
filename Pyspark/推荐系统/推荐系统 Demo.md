# 推荐系统 Demo

## 代码

	from pyspark.sql import SparkSession
	from pyspark.sql.functions import *
	from pyspark.ml.evaluation import RegressionEvaluator
	from pyspark.ml.feature import StringIndexer, IndexToString
	from pyspark.ml.recommendation import ALS
	
	if __name__ == '__main__':
	
	    spark = SparkSession.builder.appName("recomand_system").getOrCreate()
	    df = spark.read.csv("/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_8_Recommender_System/movie_ratings_df.csv"
	                        ,inferSchema=True,header=True)
	    print("原始数据：")
	    print(df.show(10,False))
	
	    #特征工程,将电影的title变为数字
	    stringIndexer = StringIndexer(inputCol="title", outputCol="title_new")
	    model = stringIndexer.fit(df)
	    indexed = model.transform(df)
	    print("将电影的title变为数字：")
	    print(indexed.show(10))
	
	    # 划分数据集
	    train,test = indexed.randomSplit([0.75,0.25])
	
	    # 构建模型: 交替最小平方 (ALS)
	    rec = ALS(maxIter=10, regParam=0.01,userCol='userId',itemCol='title_new', ratingCol='rating'
	              ,nonnegative= True, coldStartStrategy="drop")
	    rec_model= rec.fit(train)
	
	    # 评估模型
	    predicted_ratings = rec_model.transform(test)
	    evaluator = RegressionEvaluator(metricName='rmse', predictionCol='prediction', labelCol='rating')
	    rmse = evaluator.evaluate(predicted_ratings)
	    print("rmse = ", rmse) #rmse =  1.0191103160262351, 评分并不是很高，所以需要优化模型
	
	    # 推荐活动用户可能会喜欢的排名靠前的电影
	    unique_movies = indexed.select("title_new").distinct()
	    a = unique_movies.alias('a')
	    print("电影的部数为:",unique_movies.count())
	
	    #以用户id85的用户为例
	    watched_movies = indexed.filter(indexed['userId']==85).select('title_new').distinct()
	    print("用户id为85的用户看的电影部属为： ",watched_movies.count())
	    b = watched_movies.alias('b')
	
	    # 1664部电影中有287部被用户idwei85的用户评价过了，推荐其他电影.下面的代码用来确定用户id为85的用户待观看的电影title
	    total_movies = a.join(b,a.title_new == b.title_new, how='left')
	    print("左边是所有的电影的id，右边是看过的电影的id:")
	    print(total_movies.show(5,False))
	    remaining_movies = total_movies.where(col("b.title_new").isNull()).select(a.title_new).distinct()
	    remaining_movies = remaining_movies.withColumn("userId", lit(int(85))) #85为用户id
	    print("待推荐的电影总数量为：",remaining_movies.count())
	    print("用户看过的电影id：")
	    print( remaining_movies.show(10,False) )
	
	    # 推荐
	    recommendations = rec_model.transform(remaining_movies).orderBy('prediction', ascending=False)
	    print("最高预测评分的电影为：")
	    print(recommendations.show(5,False) )
	
	    # 显示电影名
	    movie_title = IndexToString(inputCol="title_new", outputCol="title", labels=model.labels)
	    final_recommendations = movie_title.transform(recommendations)
	    print("展示最终结果:")
	    print( final_recommendations.show(10,False))
	    

## 显示

	原始数据：
	+------+------------+------+
	|userId|title       |rating|
	+------+------------+------+
	|196   |Kolya (1996)|3     |
	|63    |Kolya (1996)|3     |
	|226   |Kolya (1996)|5     |
	|154   |Kolya (1996)|3     |
	|306   |Kolya (1996)|5     |
	|296   |Kolya (1996)|4     |
	|34    |Kolya (1996)|5     |
	|271   |Kolya (1996)|4     |
	|201   |Kolya (1996)|4     |
	|209   |Kolya (1996)|4     |
	+------+------------+------+
	
	
	将电影的title变为数字：
	+------+------------+------+---------+
	|userId|       title|rating|title_new|
	+------+------------+------+---------+
	|   196|Kolya (1996)|     3|    287.0|
	|    63|Kolya (1996)|     3|    287.0|
	|   226|Kolya (1996)|     5|    287.0|
	|   154|Kolya (1996)|     3|    287.0|
	|   306|Kolya (1996)|     5|    287.0|
	|   296|Kolya (1996)|     4|    287.0|
	|    34|Kolya (1996)|     5|    287.0|
	|   271|Kolya (1996)|     4|    287.0|
	|   201|Kolya (1996)|     4|    287.0|
	|   209|Kolya (1996)|     4|    287.0|
	+------+------------+------+---------+
	
	
	
	rmse =  1.0353702183819713
	电影的部数为: 1664
	用户id为85的用户看的电影部属为：  287
	
	
	
	左边是所有的电影的id，右边是看过的电影的id:
	+---------+---------+
	|title_new|title_new|
	+---------+---------+
	|305.0    |305.0    |
	|596.0    |null     |
	|299.0    |null     |
	|769.0    |null     |
	|692.0    |null     |
	+---------+---------+
	
	
	待推荐的电影总数量为： 1377
	用户看过的电影id：
	+---------+------+
	|title_new|userId|
	+---------+------+
	|596.0    |85    |
	|299.0    |85    |
	|769.0    |85    |
	|692.0    |85    |
	|934.0    |85    |
	|1051.0   |85    |
	|496.0    |85    |
	|170.0    |85    |
	|184.0    |85    |
	|576.0    |85    |
	+---------+------+
	
	
	最高预测评分的电影为：
      +---------+------+----------+
	|title_new|userId|prediction|
	+---------+------+----------+
	|892.0    |85    |5.08262   |
	|915.0    |85    |5.045919  |
	|1059.0   |85    |5.028051  |
	|1113.0   |85    |4.9901257 |
	|1120.0   |85    |4.9760666 |
	+---------+------+----------+
	
	
	展示最终结果:
	 +---------+------+----------+------------------------------------------------------------------+
	|title_new|userId|prediction|title                                                             |
	+---------+------+----------+------------------------------------------------------------------+
	|892.0    |85    |5.08262   |Double vie de Vronique, La (Double Life of Veronique, The) (1991)|
	|915.0    |85    |5.045919  |Oscar & Lucinda (1997)                                            |
	|1059.0   |85    |5.028051  |Tango Lesson, The (1997)                                          |
	|1113.0   |85    |4.9901257 |Before the Rain (Pred dozhdot) (1994)                             |
	|1120.0   |85    |4.9760666 |Crooklyn (1994)                                                   |
	|860.0    |85    |4.971077  |Naked (1993)                                                      |
	|882.0    |85    |4.785707  |Live Nude Girls (1995)                                            |
	|601.0    |85    |4.7415895 |Perfect World, A (1993)                                           |
	|963.0    |85    |4.521393  |Inspector General, The (1949)                                     |
	|691.0    |85    |4.5093317 |Some Folks Call It a Sling Blade (1993)                           |
	+---------+------+----------+------------------------------------------------------------------+