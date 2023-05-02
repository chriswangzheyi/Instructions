# QuantileDiscretizer用法
参考：https://www.jianshu.com/p/6aa0b853ae80

## 定义

QuantileDiscretizer（分位数离散化）将一列连续型的数据列转成分类型数据。通过取一个样本的数据，并将其分为大致相等的部分，设定范围。其下限为 -Infinity(负无穷大) ，上限为+Infinity(正无穷大)。

通过设置numBuckets（桶数目）来所需离散的数目。


## Demo

### 代码

	package com.sparrowrecsys
	
	import org.apache.log4j.{Level, Logger}
	import org.apache.spark.ml.feature.QuantileDiscretizer
	import org.apache.spark.sql.SparkSession
	
	
	object QuantileDiscretizerExample {
	  def main(args: Array[String]) {
	    Logger.getLogger("org").setLevel(Level.ERROR)
	    val spark = SparkSession.builder().master("local[*]").appName("QuantileDiscretizerExample").getOrCreate()
	    val sc = spark.sparkContext
	    val sqlContext = spark.sqlContext
	    import sqlContext.implicits._
	
	    val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
	    val df = sc.parallelize(data).toDF("id", "hour")
	    df.show()
	    val discretizer = new QuantileDiscretizer()
	      .setInputCol("hour")
	      .setOutputCol("result")
	      .setNumBuckets(3)
	
	    val result = discretizer.fit(df).transform(df)
	    result.show()
	
	    sc.stop()
	  }
	}

### 打印

	+---+----+
	| id|hour|
	+---+----+
	|  0|18.0|
	|  1|19.0|
	|  2| 8.0|
	|  3| 5.0|
	|  4| 2.2|
	+---+----+
	
	+---+----+------+
	| id|hour|result|
	+---+----+------+
	|  0|18.0|   2.0|
	|  1|19.0|   2.0|
	|  2| 8.0|   1.0|
	|  3| 5.0|   1.0|
	|  4| 2.2|   0.0|
	+---+----+------+
	
## Demo2

### 代码

	package com.sparrowrecsys
	
	import org.apache.log4j.{Level, Logger}
	import org.apache.spark.ml.feature.QuantileDiscretizer
	import org.apache.spark.sql.SparkSession
	
	
	object QuantileDiscretizerExample {
	  def main(args: Array[String]) {
	    Logger.getLogger("org").setLevel(Level.ERROR)
	    val spark = SparkSession.builder().master("local[*]").appName("QuantileDiscretizerExample").getOrCreate()
	    val sc = spark.sparkContext
	    val sqlContext = spark.sqlContext
	    import sqlContext.implicits._
	
	    val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2), (5, 2.2), (6, 2.2), (7, 2.2), (8, 18.0), (9, 118.0), (111, 118.0), (112, 119.0))
	    val df = sc.parallelize(data).toDF("id", "hour")
	    df.show()
	    val discretizer = new QuantileDiscretizer()
	      .setInputCol("hour")
	      .setOutputCol("result")
	      .setNumBuckets(4)
	
	    val result = discretizer.fit(df).transform(df)
	    result.show()
	
	    sc.stop()
	  }
	}

### 打印

	+---+-----+
	| id| hour|
	+---+-----+
	|  0| 18.0|
	|  1| 19.0|
	|  2|  8.0|
	|  3|  5.0|
	|  4|  2.2|
	|  5|  2.2|
	|  6|  2.2|
	|  7|  2.2|
	|  8| 18.0|
	|  9|118.0|
	|111|118.0|
	|112|119.0|
	+---+-----+
	
	+---+-----+------+
	| id| hour|result|
	+---+-----+------+
	|  0| 18.0|   2.0|
	|  1| 19.0|   3.0|
	|  2|  8.0|   2.0|
	|  3|  5.0|   1.0|
	|  4|  2.2|   1.0|
	|  5|  2.2|   1.0|
	|  6|  2.2|   1.0|
	|  7|  2.2|   1.0|
	|  8| 18.0|   2.0|
	|  9|118.0|   3.0|
	|111|118.0|   3.0|
	|112|119.0|   3.0|
	+---+-----+------+


## Demo3

### 代码

	package com.sparrowrecsys
	
	import org.apache.log4j.{Level, Logger}
	import org.apache.spark.ml.feature.QuantileDiscretizer
	import org.apache.spark.sql.SparkSession
	
	
	object QuantileDiscretizerExample {
	  def main(args: Array[String]) {
	    Logger.getLogger("org").setLevel(Level.ERROR)
	    val spark = SparkSession.builder().master("local[*]").appName("QuantileDiscretizerExample").getOrCreate()
	    val sc = spark.sparkContext
	    val sqlContext = spark.sqlContext
	    import sqlContext.implicits._
	
	    val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2), (5, 2.2), (6, 2.2), (7, 2.2), (8, 18.0), (9, 118.0), (111, 118.0), (112, 119.0),(113, 119.0))
	    val df = sc.parallelize(data).toDF("id", "hour")
	    df.show()
	    val discretizer = new QuantileDiscretizer()
	      .setInputCol("hour")
	      .setOutputCol("result")
	      .setNumBuckets(7)
	
	    val result = discretizer.fit(df).transform(df)
	    result.show()
	
	    sc.stop()
	  }
	}


### 打印

	+---+-----+
	| id| hour|
	+---+-----+
	|  0| 18.0|
	|  1| 19.0|
	|  2|  8.0|
	|  3|  5.0|
	|  4|  2.2|
	|  5|  2.2|
	|  6|  2.2|
	|  7|  2.2|
	|  8| 18.0|
	|  9|118.0|
	|111|118.0|
	|112|119.0|
	|113|119.0|
	+---+-----+
	
	+---+-----+------+
	| id| hour|result|
	+---+-----+------+
	|  0| 18.0|   3.0|
	|  1| 19.0|   3.0|
	|  2|  8.0|   2.0|
	|  3|  5.0|   1.0|
	|  4|  2.2|   1.0|
	|  5|  2.2|   1.0|
	|  6|  2.2|   1.0|
	|  7|  2.2|   1.0|
	|  8| 18.0|   3.0|
	|  9|118.0|   4.0|
	|111|118.0|   4.0|
	|112|119.0|   5.0|
	|113|119.0|   5.0|
	+---+-----+------+
