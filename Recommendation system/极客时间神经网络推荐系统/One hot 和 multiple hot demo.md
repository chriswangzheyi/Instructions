# One hot 和 multiple hot demo

## 代码

	package com.sparrowrecsys.offline.spark.featureeng
	
	import org.apache.log4j.{Level, Logger}
	import org.apache.spark.{SparkConf, sql}
	import org.apache.spark.ml.{Pipeline, PipelineStage}
	import org.apache.spark.ml.feature._
	import org.apache.spark.sql.expressions.UserDefinedFunction
	import org.apache.spark.sql.{DataFrame, SparkSession}
	import org.apache.spark.sql.functions._
	
	object FeatureEngineering {
	  /**
	   * One-hot encoding example function
	   * @param samples movie samples dataframe
	   */
	  def oneHotEncoderExample(samples:DataFrame): Unit ={
	    val samplesWithIdNumber = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))
	
	    val oneHotEncoder = new OneHotEncoderEstimator()
	      .setInputCols(Array("movieIdNumber"))
	      .setOutputCols(Array("movieIdVector"))
	      .setDropLast(false)
	
	    val oneHotEncoderSamples = oneHotEncoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
	    oneHotEncoderSamples.printSchema()
	    oneHotEncoderSamples.show(10)
	  }
	
	  val array2vec: UserDefinedFunction = udf { (a: Seq[Int], length: Int) => org.apache.spark.ml.linalg.Vectors.sparse(length, a.sortWith(_ < _).toArray, Array.fill[Double](a.length)(1.0)) }
	
	  /**
	   * Multi-hot encoding example function
	   * @param samples movie samples dataframe
	   */
	  def multiHotEncoderExample(samples:DataFrame): Unit ={
	    val samplesWithGenre = samples.select(col("movieId"), col("title"),explode(split(col("genres"), "\\|").cast("array<string>")).as("genre"))
	    val genreIndexer = new StringIndexer().setInputCol("genre").setOutputCol("genreIndex")
	
	    val stringIndexerModel : StringIndexerModel = genreIndexer.fit(samplesWithGenre)
	
	    val genreIndexSamples = stringIndexerModel.transform(samplesWithGenre)
	      .withColumn("genreIndexInt", col("genreIndex").cast(sql.types.IntegerType))
	
	    val indexSize = genreIndexSamples.agg(max(col("genreIndexInt"))).head().getAs[Int](0) + 1
	
	    val processedSamples =  genreIndexSamples
	      .groupBy(col("movieId")).agg(collect_list("genreIndexInt").as("genreIndexes"))
	        .withColumn("indexSize", typedLit(indexSize))
	
	    val finalSample = processedSamples.withColumn("vector", array2vec(col("genreIndexes"),col("indexSize")))
	    finalSample.printSchema()
	    finalSample.show(10)
	  }
	
	  val double2vec: UserDefinedFunction = udf { (value: Double) => org.apache.spark.ml.linalg.Vectors.dense(value) }
	
	  /**
	   * Process rating samples
	   * @param samples rating samples
	   */
	  def ratingFeatures(samples:DataFrame): Unit ={
	    samples.printSchema()
	    samples.show(10)
	
	    //calculate average movie rating score and rating count
	    val movieFeatures = samples.groupBy(col("movieId"))
	      .agg(count(lit(1)).as("ratingCount"),
	        avg(col("rating")).as("avgRating"),
	        variance(col("rating")).as("ratingVar"))
	        .withColumn("avgRatingVec", double2vec(col("avgRating")))
	
	    movieFeatures.show(10)
	
	    //bucketing
	    val ratingCountDiscretizer = new QuantileDiscretizer()
	      .setInputCol("ratingCount")
	      .setOutputCol("ratingCountBucket")
	      .setNumBuckets(100)
	
	    //Normalization
	    val ratingScaler = new MinMaxScaler()
	      .setInputCol("avgRatingVec")
	      .setOutputCol("scaleAvgRating")
	
	    val pipelineStage: Array[PipelineStage] = Array(ratingCountDiscretizer, ratingScaler)
	    val featurePipeline = new Pipeline().setStages(pipelineStage)
	
	    val movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
	    movieProcessedFeatures.show(10)
	  }
	
	  def main(args: Array[String]): Unit = {
	    Logger.getLogger("org").setLevel(Level.ERROR)
	
	    val conf = new SparkConf()
	      .setMaster("local")
	      .setAppName("featureEngineering")
	      .set("spark.submit.deployMode", "client")
	
	    val spark = SparkSession.builder.config(conf).getOrCreate()
	
	
	
	    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
	    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
	    println("Raw Movie Samples:")
	    movieSamples.printSchema()
	    movieSamples.show(10)
	
	    println("OneHotEncoder Example:")
	    oneHotEncoderExample(movieSamples)
	
	    println("MultiHotEncoder Example:")
	    multiHotEncoderExample(movieSamples)
	
	    println("Numerical features Example:")
	    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
	    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
	    ratingFeatures(ratingSamples)
	
	  }
	}


## 输出 


	Raw Movie Samples:
	root
	 |-- movieId: string (nullable = true)
	 |-- title: string (nullable = true)
	 |-- genres: string (nullable = true)
	
	+-------+--------------------+--------------------+
	|movieId|               title|              genres|
	+-------+--------------------+--------------------+
	|      1|    Toy Story (1995)|Adventure|Animati...|
	|      2|      Jumanji (1995)|Adventure|Childre...|
	|      3|Grumpier Old Men ...|      Comedy|Romance|
	|      4|Waiting to Exhale...|Comedy|Drama|Romance|
	|      5|Father of the Bri...|              Comedy|
	|      6|         Heat (1995)|Action|Crime|Thri...|
	|      7|      Sabrina (1995)|      Comedy|Romance|
	|      8| Tom and Huck (1995)|  Adventure|Children|
	|      9| Sudden Death (1995)|              Action|
	|     10|    GoldenEye (1995)|Action|Adventure|...|
	+-------+--------------------+--------------------+
	only showing top 10 rows
	
	OneHotEncoder Example:
	root
	 |-- movieId: string (nullable = true)
	 |-- title: string (nullable = true)
	 |-- genres: string (nullable = true)
	 |-- movieIdNumber: integer (nullable = true)
	 |-- movieIdVector: vector (nullable = true)
	
	+-------+--------------------+--------------------+-------------+-----------------+
	|movieId|               title|              genres|movieIdNumber|    movieIdVector|
	+-------+--------------------+--------------------+-------------+-----------------+
	|      1|    Toy Story (1995)|Adventure|Animati...|            1| (1001,[1],[1.0])|
	|      2|      Jumanji (1995)|Adventure|Childre...|            2| (1001,[2],[1.0])|
	|      3|Grumpier Old Men ...|      Comedy|Romance|            3| (1001,[3],[1.0])|
	|      4|Waiting to Exhale...|Comedy|Drama|Romance|            4| (1001,[4],[1.0])|
	|      5|Father of the Bri...|              Comedy|            5| (1001,[5],[1.0])|
	|      6|         Heat (1995)|Action|Crime|Thri...|            6| (1001,[6],[1.0])|
	|      7|      Sabrina (1995)|      Comedy|Romance|            7| (1001,[7],[1.0])|
	|      8| Tom and Huck (1995)|  Adventure|Children|            8| (1001,[8],[1.0])|
	|      9| Sudden Death (1995)|              Action|            9| (1001,[9],[1.0])|
	|     10|    GoldenEye (1995)|Action|Adventure|...|           10|(1001,[10],[1.0])|
	+-------+--------------------+--------------------+-------------+-----------------+
	only showing top 10 rows
	
	MultiHotEncoder Example:
	root
	 |-- movieId: string (nullable = true)
	 |-- genreIndexes: array (nullable = true)
	 |    |-- element: integer (containsNull = true)
	 |-- indexSize: integer (nullable = false)
	 |-- vector: vector (nullable = true)
	
	+-------+------------+---------+--------------------------------+
	|movieId|genreIndexes|indexSize|vector                          |
	+-------+------------+---------+--------------------------------+
	|296    |[1, 5, 0, 3]|19       |(19,[0,1,3,5],[1.0,1.0,1.0,1.0])|
	|467    |[1]         |19       |(19,[1],[1.0])                  |
	|675    |[4, 0, 3]   |19       |(19,[0,3,4],[1.0,1.0,1.0])      |
	|691    |[1, 2]      |19       |(19,[1,2],[1.0,1.0])            |
	|829    |[1, 10, 14] |19       |(19,[1,10,14],[1.0,1.0,1.0])    |
	|125    |[1]         |19       |(19,[1],[1.0])                  |
	|451    |[0, 8, 2]   |19       |(19,[0,2,8],[1.0,1.0,1.0])      |
	|800    |[0, 8, 16]  |19       |(19,[0,8,16],[1.0,1.0,1.0])     |
	|853    |[0]         |19       |(19,[0],[1.0])                  |
	|944    |[0]         |19       |(19,[0],[1.0])                  |
	+-------+------------+---------+--------------------------------+
	only showing top 10 rows
	
	Numerical features Example:
	root
	 |-- userId: string (nullable = true)
	 |-- movieId: string (nullable = true)
	 |-- rating: string (nullable = true)
	 |-- timestamp: string (nullable = true)
	
	+------+-------+------+----------+
	|userId|movieId|rating| timestamp|
	+------+-------+------+----------+
	|     1|      2|   3.5|1112486027|
	|     1|     29|   3.5|1112484676|
	|     1|     32|   3.5|1112484819|
	|     1|     47|   3.5|1112484727|
	|     1|     50|   3.5|1112484580|
	|     1|    112|   3.5|1094785740|
	|     1|    151|   4.0|1094785734|
	|     1|    223|   4.0|1112485573|
	|     1|    253|   4.0|1112484940|
	|     1|    260|   4.0|1112484826|
	+------+-------+------+----------+
	only showing top 10 rows
	
	+-------+-----------+------------------+------------------+--------------------+
	|movieId|ratingCount|         avgRating|         ratingVar|        avgRatingVec|
	+-------+-----------+------------------+------------------+--------------------+
	|    296|      14616| 4.165606185002737|0.9615737413069363| [4.165606185002737]|
	|    467|        174|3.4367816091954024|1.5075410271742742|[3.4367816091954024]|
	|    829|        402|2.6243781094527363|1.4982072182727266|[2.6243781094527363]|
	|    691|        254|3.1161417322834644|1.0842838691606236|[3.1161417322834644]|
	|    675|          6|2.3333333333333335|0.6666666666666667|[2.3333333333333335]|
	|    125|        788| 3.713197969543147|0.8598255922703314| [3.713197969543147]|
	|    800|       1609|4.0447482908638905|0.8325734596130598|[4.0447482908638905]|
	|    944|        259|3.8262548262548264|0.8534165394630507|[3.8262548262548264]|
	|    853|         20|               3.5| 1.526315789473684|               [3.5]|
	|    451|        159|  3.00314465408805|0.7800533397022527|  [3.00314465408805]|
	+-------+-----------+------------------+------------------+--------------------+
	only showing top 10 rows
	
	+-------+-----------+------------------+------------------+--------------------+-----------------+--------------------+
	|movieId|ratingCount|         avgRating|         ratingVar|        avgRatingVec|ratingCountBucket|      scaleAvgRating|
	+-------+-----------+------------------+------------------+--------------------+-----------------+--------------------+
	|    296|      14616| 4.165606185002737|0.9615737413069363| [4.165606185002737]|             99.0|[0.9170998054196597]|
	|    467|        174|3.4367816091954024|1.5075410271742742|[3.4367816091954024]|             38.0|[0.7059538707722662]|
	|    829|        402|2.6243781094527363|1.4982072182727266|[2.6243781094527363]|             54.0|[0.47059449629732...|
	|    691|        254|3.1161417322834644|1.0842838691606236|[3.1161417322834644]|             45.0|[0.6130620985364005]|
	|    675|          6|2.3333333333333335|0.6666666666666667|[2.3333333333333335]|              4.0|[0.3862766462716174]|
	|    125|        788| 3.713197969543147|0.8598255922703314| [3.713197969543147]|             67.0|[0.7860337592595665]|
	|    800|       1609|4.0447482908638905|0.8325734596130598|[4.0447482908638905]|             79.0|[0.8820863689021069]|
	|    944|        259|3.8262548262548264|0.8534165394630507|[3.8262548262548264]|             46.0|[0.8187871768460152]|
	|    853|         20|               3.5| 1.526315789473684|               [3.5]|             12.0|[0.7242687117592825]|
	|    451|        159|  3.00314465408805|0.7800533397022527|  [3.00314465408805]|             37.0|[0.5803259992335382]|
	+-------+-----------+------------------+------------------+--------------------+-----------------+--------------------+
	only showing top 10 rows



## 解释

### one hot输出解释

movieIdVector将1001部电影，分解为1001维的向量

### 