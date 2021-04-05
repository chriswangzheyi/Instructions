# 自然语言处理 Demo 

## 代码

	from pyspark.ml.classification import LogisticRegression
	from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, VectorAssembler
	from pyspark.sql import SparkSession
	from pyspark.sql.functions import rand, length, udf, col
	from pyspark.sql.types import IntegerType
	
	if __name__ == '__main__':
	
	    spark = SparkSession.builder.appName('nlp').getOrCreate()
	    text_df = spark.read.csv('/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_9_NLP/Movie_reviews.csv'
	                             ,inferSchema=True,header=True)
	
	    print(text_df.orderBy(rand()).show(10,False))
	
	    #筛选出正确被标记的记录
	    text_df=text_df.filter(((text_df.Sentiment == '1') | (text_df.Sentiment == '0')))
	    print("被正确标记的数据条数：")
	    print(text_df.count())
	
	    # 将sentiment变为label
	    text_df = text_df.withColumn("Label", text_df.Sentiment.cast('float')).drop('Sentiment')
	    print("加label后的数据：")
	    print(text_df.orderBy(rand()).show(10, False))
	
	    # 加一列捕获评论的长度
	    text_df = text_df.withColumn("length",length(text_df['Review']))
	    print("加length后的数据：")
	    print(text_df.orderBy(rand()).show(10, False))
	
	    #删减停用词
	    tokenization = Tokenizer(inputCol='Review',outputCol='tokens')
	    tokenized_df = tokenization.transform(text_df)
	    stopword_removal = StopWordsRemover(inputCol='tokens',outputCol='refined_tokens')
	    refined_text_df = stopword_removal.transform(tokenized_df)
	    print("删减停用词后：")
	    print(refined_text_df.show(10,False))
	
	    # 标记数量
	    len_udf = udf(lambda s: len(s), IntegerType())
	    refined_text_df = refined_text_df.withColumn("token_count",len_udf(col('refined_tokens')))
	    print("加上标记数量后：")
	    print(refined_text_df.orderBy(rand()).show() )
	
	    # 特征向量化
	    count_vec = CountVectorizer(inputCol='refined_tokens', outputCol='features')
	    cv_text_df = count_vec.fit(refined_text_df).transform(refined_text_df)
	    model_text_df = cv_text_df.select(['features','token_count','Label'])
	    print("特征向量化后：")
	    print(model_text_df.show(10,False))
	
	    # 创建输入特征
	    df_assembler = VectorAssembler(inputCols=['features','token_count'], outputCol='features_vec')
	    model_text_df = df_assembler.transform(model_text_df)
	    print("model输入特征")
	    print(model_text_df.show(10,False))
	
	    # 数据分组
	    training_df, test_df = model_text_df.randomSplit([0.75,0.25])
	
	
	    # 构建模型,用逻辑回归
	    log_reg = LogisticRegression(featuresCol='features_vec', labelCol='Label').fit(training_df)
	
	    # 检视模型
	    results = log_reg.evaluate(test_df).predictions
	    print("检视模型：")
	    print(results.show())
	
	    # 评价模型
	    tp = results[(results.Label == 1) & (results.prediction == 1)].count()
	    tn = results[(results.Label == 0) & (results.prediction == 0)].count()
	    fp = results[(results.Label == 0) & (results.prediction == 1)].count()
	    fn = results[(results.Label == 1) & (results.prediction == 0)].count()
	
	    recall = float(tp/(tp+fn))
	    print("recall =", recall)
	    precision = float(tp/(tp+fp))
	    print("precision =", precision)
	    accuracy = float( (tp+tn)/results.count())
	    print("accuracy =",accuracy)
	    
## 显示

	+------------------------------------------------------------------------+---------+
	|Review                                                                  |Sentiment|
	+------------------------------------------------------------------------+---------+
	|"Anyway, thats why I love "" Brokeback Mountain."                       |1        |
	|He's like,'YEAH I GOT ACNE AND I LOVE BROKEBACK MOUNTAIN '..            |1        |
	|the story of Harry Potter is a deep and profound one, and I love Harry P|1        |
	|Mission Impossible 3 was an awesome film, the characterization was great|1        |
	|Harry Potter is AWESOME I don't care if anyone says differently!..      |1        |
	|brokeback mountain is such a beautiful movie.                           |1        |
	|I love Harry Potter, Twilight, Series of Unfortunate Events, and tons mo|1        |
	|Harry Potter is AWESOME I don't care if anyone says differently!..      |1        |
	|I thought Brokeback Mountain was beautiful too, and I cried too, which I|1        |
	|"I liked the first "" Mission Impossible."                              |1        |
	+------------------------------------------------------------------------+---------+
	
	
	被正确标记的数据条数：
	6990
	
	加label后的数据：
	+------------------------------------------------------------------------+-----+
	|Review                                                                  |Label|
	+------------------------------------------------------------------------+-----+
	|He's like,'YEAH I GOT ACNE AND I LOVE BROKEBACK MOUNTAIN '..            |1.0  |
	|I haven't seen Crash-and everyone tells me it's fabulous, but I just lov|1.0  |
	|This quiz sucks and Harry Potter sucks ok bye..                         |0.0  |
	|I hate Harry Potter, it's retarted, gay and stupid and there's only one |0.0  |
	|The Da Vinci Code is awesome!!                                          |1.0  |
	|Love luv lubb the Da Vinci Code!                                        |1.0  |
	|Brokeback Mountain was so awesome.                                      |1.0  |
	|DA VINCI CODE IS AWESOME!!                                              |1.0  |
	|Mission Impossible 3 is actually pretty awesome as far as mindless actio|1.0  |
	|He's like,'YEAH I GOT ACNE AND I LOVE BROKEBACK MOUNTAIN '..            |1.0  |
	+------------------------------------------------------------------------+-----+
	
	
	加length后的数据：
	+------------------------------------------------------------------------+-----+------+
	|Review                                                                  |Label|length|
	+------------------------------------------------------------------------+-----+------+
	|Combining the opinion / review from Gary and Gin Zen, The Da Vinci Code |0.0  |71    |
	|I like Mission Impossible movies because you never know who's on the rig|1.0  |72    |
	|I think I hate Harry Potter because it outshines much better reading mat|0.0  |72    |
	|He's like,'YEAH I GOT ACNE AND I LOVE BROKEBACK MOUNTAIN '..            |1.0  |60    |
	|i love being a sentry for mission impossible and a station for bonkers. |1.0  |71    |
	|we're gonna like watch Mission Impossible or Hoot.(                     |1.0  |51    |
	|the story of Harry Potter is a deep and profound one, and I love Harry P|1.0  |72    |
	|Harry Potter is AWESOME I don't care if anyone says differently!..      |1.0  |66    |
	|BROKEBACK MOUNTAIN SUCKS....                                            |0.0  |28    |
	|I want to be here because I love Harry Potter, and I really want a place|1.0  |72    |
	+------------------------------------------------------------------------+-----+------+
	
	
	删减停用词后：
	+------------------------------------------------------------------------+-----+------+----------------------------------------------------------------------------------------+-------------------------------------------------------------+
	|Review                                                                  |Label|length|tokens                                                                                  |refined_tokens                                               |
	+------------------------------------------------------------------------+-----+------+----------------------------------------------------------------------------------------+-------------------------------------------------------------+
	|The Da Vinci Code book is just awesome.                                 |1.0  |39    |[the, da, vinci, code, book, is, just, awesome.]                                        |[da, vinci, code, book, awesome.]                            |
	|this was the first clive cussler i've ever read, but even books like Rel|1.0  |72    |[this, was, the, first, clive, cussler, i've, ever, read,, but, even, books, like, rel] |[first, clive, cussler, ever, read,, even, books, like, rel] |
	|i liked the Da Vinci Code a lot.                                        |1.0  |32    |[i, liked, the, da, vinci, code, a, lot.]                                               |[liked, da, vinci, code, lot.]                               |
	|i liked the Da Vinci Code a lot.                                        |1.0  |32    |[i, liked, the, da, vinci, code, a, lot.]                                               |[liked, da, vinci, code, lot.]                               |
	|I liked the Da Vinci Code but it ultimatly didn't seem to hold it's own.|1.0  |72    |[i, liked, the, da, vinci, code, but, it, ultimatly, didn't, seem, to, hold, it's, own.]|[liked, da, vinci, code, ultimatly, seem, hold, own.]        |
	|that's not even an exaggeration ) and at midnight we went to Wal-Mart to|1.0  |72    |[that's, not, even, an, exaggeration, ), and, at, midnight, we, went, to, wal-mart, to] |[even, exaggeration, ), midnight, went, wal-mart]            |
	|I loved the Da Vinci Code, but now I want something better and different|1.0  |72    |[i, loved, the, da, vinci, code,, but, now, i, want, something, better, and, different] |[loved, da, vinci, code,, want, something, better, different]|
	|i thought da vinci code was great, same with kite runner.               |1.0  |57    |[i, thought, da, vinci, code, was, great,, same, with, kite, runner.]                   |[thought, da, vinci, code, great,, kite, runner.]            |
	|The Da Vinci Code is actually a good movie...                           |1.0  |45    |[the, da, vinci, code, is, actually, a, good, movie...]                                 |[da, vinci, code, actually, good, movie...]                  |
	|I thought the Da Vinci Code was a pretty good book.                     |1.0  |51    |[i, thought, the, da, vinci, code, was, a, pretty, good, book.]                         |[thought, da, vinci, code, pretty, good, book.]              |
	+------------------------------------------------------------------------+-----+------+----------------------------------------------------------------------------------------+-------------------------------------------------------------+
	
	
	加上标记数量后：
	 +--------------------+-----+------+--------------------+--------------------+-----------+
	|              Review|Label|length|              tokens|      refined_tokens|token_count|
	+--------------------+-----+------+--------------------+--------------------+-----------+
	|Because I would l...|  1.0|    72|[because, i, woul...|[like, make, frie...|          6|
	|My dad's being st...|  0.0|    49|[my, dad's, being...|[dad's, stupid, b...|          4|
	|I, too, like Harr...|  1.0|    27|[i,, too,, like, ...|[i,, too,, like, ...|          5|
	|Brokeback Mountai...|  0.0|    40|[brokeback, mount...|[brokeback, mount...|          4|
	|. Brokeback Mount...|  0.0|    27|[., brokeback, mo...|[., brokeback, mo...|          4|
	|I like Mission Im...|  1.0|    72|[i, like, mission...|[like, mission, i...|          7|
	|I am going to sta...|  1.0|    72|[i, am, going, to...|[going, start, re...|          6|
	|I either LOVE Bro...|  1.0|    71|[i, either, love,...|[either, love, br...|          7|
	|And I hate Harry ...|  0.0|    34|[and, i, hate, ha...|[hate, harry, pot...|          5|
	|He's like,'YEAH I...|  1.0|    60|[he's, like,'yeah...|[like,'yeah, got,...|          7|
	|I hate playing Mi...|  0.0|    48|[i, hate, playing...|[hate, playing, m...|          7|
	|Da Vinci Code sucks.|  0.0|    20|[da, vinci, code,...|[da, vinci, code,...|          4|
	|Love luv lubb the...|  1.0|    32|[love, luv, lubb,...|[love, luv, lubb,...|          6|
	|Is it just me, or...|  0.0|    44|[is, it, just, me...|[me,, harry, pott...|          4|
	|"I liked the firs...|  1.0|    42|["i, liked, the, ...|["i, liked, first...|          6|
	|Brokeback mountai...|  1.0|    35|[brokeback, mount...|[brokeback, mount...|          3|
	|I love Harry Pott...|  1.0|    21|[i, love, harry, ...|[love, harry, pot...|          3|
	|The Da Vinci Code...|  1.0|    57|[the, da, vinci, ...|[da, vinci, code,...|          7|
	|I hate Harry Potter.|  0.0|    20|[i, hate, harry, ...|[hate, harry, pot...|          3|
	|Da Vinci Code suc...|  0.0|    22|[da, vinci, code,...|[da, vinci, code,...|          4|
	+--------------------+-----+------+--------------------+--------------------+-----------+
	
	
	特征向量化后：
	+----------------------------------------------------------------------------------+-----------+-----+
	|features                                                                          |token_count|Label|
	+----------------------------------------------------------------------------------+-----------+-----+
	|(2302,[0,1,4,43,236],[1.0,1.0,1.0,1.0,1.0])                                       |5          |1.0  |
	|(2302,[11,51,229,237,275,742,824,1087,1250],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|9          |1.0  |
	|(2302,[0,1,4,53,356],[1.0,1.0,1.0,1.0,1.0])                                       |5          |1.0  |
	|(2302,[0,1,4,53,356],[1.0,1.0,1.0,1.0,1.0])                                       |5          |1.0  |
	|(2302,[0,1,4,53,655,1339,1427,1449],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])            |8          |1.0  |
	|(2302,[46,229,271,1150,1990,2203],[1.0,1.0,1.0,1.0,1.0,1.0])                      |6          |1.0  |
	|(2302,[0,1,22,30,111,219,389,535],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])              |8          |1.0  |
	|(2302,[0,1,4,228,1258,1716,2263],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])                   |7          |1.0  |
	|(2302,[0,1,4,33,226,258],[1.0,1.0,1.0,1.0,1.0,1.0])                               |6          |1.0  |
	|(2302,[0,1,4,223,226,228,262],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])                      |7          |1.0  |
	+----------------------------------------------------------------------------------+-----------+-----+


	model输入特征
	+----------------------------------------------------------------------------------+-----------+-----+-------------------------------------------------------------------------------------------+
	|features                                                                          |token_count|Label|features_vec                                                                               |
	+----------------------------------------------------------------------------------+-----------+-----+-------------------------------------------------------------------------------------------+
	|(2302,[0,1,4,43,236],[1.0,1.0,1.0,1.0,1.0])                                       |5          |1.0  |(2303,[0,1,4,43,236,2302],[1.0,1.0,1.0,1.0,1.0,5.0])                                       |
	|(2302,[11,51,229,237,275,742,824,1087,1250],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])|9          |1.0  |(2303,[11,51,229,237,275,742,824,1087,1250,2302],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0])|
	|(2302,[0,1,4,53,356],[1.0,1.0,1.0,1.0,1.0])                                       |5          |1.0  |(2303,[0,1,4,53,356,2302],[1.0,1.0,1.0,1.0,1.0,5.0])                                       |
	|(2302,[0,1,4,53,356],[1.0,1.0,1.0,1.0,1.0])                                       |5          |1.0  |(2303,[0,1,4,53,356,2302],[1.0,1.0,1.0,1.0,1.0,5.0])                                       |
	|(2302,[0,1,4,53,655,1339,1427,1449],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])            |8          |1.0  |(2303,[0,1,4,53,655,1339,1427,1449,2302],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,8.0])            |
	|(2302,[46,229,271,1150,1990,2203],[1.0,1.0,1.0,1.0,1.0,1.0])                      |6          |1.0  |(2303,[46,229,271,1150,1990,2203,2302],[1.0,1.0,1.0,1.0,1.0,1.0,6.0])                      |
	|(2302,[0,1,22,30,111,219,389,535],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])              |8          |1.0  |(2303,[0,1,22,30,111,219,389,535,2302],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,8.0])              |
	|(2302,[0,1,4,228,1258,1716,2263],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])                   |7          |1.0  |(2303,[0,1,4,228,1258,1716,2263,2302],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,7.0])                   |
	|(2302,[0,1,4,33,226,258],[1.0,1.0,1.0,1.0,1.0,1.0])                               |6          |1.0  |(2303,[0,1,4,33,226,258,2302],[1.0,1.0,1.0,1.0,1.0,1.0,6.0])                               |
	|(2302,[0,1,4,223,226,228,262],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])                      |7          |1.0  |(2303,[0,1,4,223,226,228,262,2302],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,7.0])                      |
	+----------------------------------------------------------------------------------+-----------+-----+-------------------------------------------------------------------------------------------+
	
	
	检视模型：
	+--------------------+-----------+-----+--------------------+--------------------+--------------------+----------+
	|            features|token_count|Label|        features_vec|       rawPrediction|         probability|prediction|
	+--------------------+-----------+-----+--------------------+--------------------+--------------------+----------+
	|(2302,[0,1,4,5,64...|          6|  1.0|(2303,[0,1,4,5,64...|[-23.278943137576...|[7.76396288796379...|       1.0|
	|(2302,[0,1,4,5,22...|          9|  1.0|(2303,[0,1,4,5,22...|[-3.5899324688878...|[0.02685888382178...|       1.0|
	|(2302,[0,1,4,11,1...|          6|  0.0|(2303,[0,1,4,11,1...|[3.19703981202368...|[0.96072272800418...|       0.0|
	|(2302,[0,1,4,11,1...|          6|  0.0|(2303,[0,1,4,11,1...|[3.19703981202368...|[0.96072272800418...|       0.0|
	|(2302,[0,1,4,11,2...|         10|  0.0|(2303,[0,1,4,11,2...|[20.8072506979885...|[0.99999999908055...|       0.0|
	|(2302,[0,1,4,12,1...|          8|  1.0|(2303,[0,1,4,12,1...|[-12.439786034403...|[3.95792764226580...|       1.0|
	|(2302,[0,1,4,12,2...|          9|  1.0|(2303,[0,1,4,12,2...|[-42.316729028987...|[4.18868875183369...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	|(2302,[0,1,4,12,3...|          5|  1.0|(2303,[0,1,4,12,3...|[-20.919559481745...|[8.21770889821630...|       1.0|
	+--------------------+-----------+-----+--------------------+--------------------+--------------------+----------+
	
	
	
	recall = 0.9896907216494846
	precision = 0.9687184661957619
	accuracy = 0.9766381766381766
