# Spark Streaming 集成 Spark SQL Demo

## 

	import org.apache.spark.SparkConf
	import org.apache.spark.sql.SparkSession
	import org.apache.spark.storage.StorageLevel
	import org.apache.spark.streaming.{Seconds, StreamingContext}
	
	object MyNetworkWordCountDataFrame {
	  def main(args: Array[String]): Unit = {
	    //创建一个Context对象: StreamingContext  (SparkContext, SQLContext)
	    //指定批处理的时间间隔
	    val conf = new SparkConf().setAppName("MyNetworkWordCount").setMaster("local[2]")
	    val ssc = new StreamingContext(conf, Seconds(5))
	
	    //创建一个DStream，处理数据
	    val lines = ssc.socketTextStream("192.168.2.101", 1234, StorageLevel.MEMORY_AND_DISK_SER)
	
	    //执行wordcount
	    val words = lines.flatMap(_.split(" "))
	
	    //使用Spark SQL来查询Spark Streaming处理的数据
	    words.foreachRDD { rdd =>
	      //使用单列模式，创建SparkSession对象
	      val spark = SparkSession.builder.config(rdd.sparkContext.getConf).getOrCreate()
	
	      import spark.implicits._
	      // 将RDD[String]转换为DataFrame
	      val wordsDataFrame = rdd.toDF("word")
	
	      // 创建临时视图
	      wordsDataFrame.createOrReplaceTempView("words")
	
	      // 执行SQL
	      val wordCountsDataFrame =   spark.sql("select word, count(*) as total from words group by word")
	      wordCountsDataFrame.show()
	    }
	
	    //启动StreamingContext
	    ssc.start()
	
	    //等待计算完成
	    ssc.awaitTermination()
	  }
	}
	
## 测试结果

### netcat端

输入

	i love you love they

### 客户端

统计结果如下

	+----+-----+
	|word|total|
	+----+-----+
	| you|    1|
	|love|    2|
	|they|    1|
	|   i|    1|
	+----+-----+