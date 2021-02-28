#08 Hudi 更新和查询

## 代码

### HudiTest

	package com.wzy.test
	
	import com.wzy.entity.WzyEntity
	import com.wzy.util.JsonUtil
	import org.apache.hudi.{DataSourceReadOptions, DataSourceWriteOptions}
	import org.apache.spark.SparkConf
	import org.apache.spark.sql.{SaveMode, SparkSession}
	
	object HudiTest {
	
	  def main(args: Array[String]): Unit = {
	    System.setProperty("HADOOP_USER_NAME", "root")
	    val sparkConf = new SparkConf().setAppName("HudiTest")
	      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
	      .setMaster("local[*]")
	    val sparkSession = SparkSession.builder().config(sparkConf).enableHiveSupport().getOrCreate()
	    val ssc = sparkSession.sparkContext
	
	    ssc.hadoopConfiguration.set("dfs.client.use.datanode.hostname","true");
	    //insertData(sparkSession)
	    //updateData(sparkSession)
	    //queryData(sparkSession)
	    //incrementalQuery(sparkSession)
	    timeQuery(sparkSession)
	  }
	
	  /**
	   * @param sparkSession
	   * @return
	   */
	  def insertData(sparkSession: SparkSession) = {
	    import org.apache.spark.sql.functions._
	    import sparkSession.implicits._
	
	    val commitTime = System.currentTimeMillis().toString //生成提交时间
	
	    val df = sparkSession.read.text("/test/test11")
	      .mapPartitions(partitions => {
	        partitions.map(item => {
	          val jsonObject = JsonUtil.getJsonData(item.getString(0))
	          WzyEntity(jsonObject.getIntValue("uid"), jsonObject.getString("uname"), jsonObject.getString("dt"))
	        })
	      })
	
	    val result = df.withColumn("ts", lit(commitTime)) //添加ts 时间戳列
	      .withColumn("uuid", col("uid"))
	      .withColumn("hudipart", col("dt")) //增加hudi分区列
	
	    result.write.format("org.apache.hudi")
	      .option("hoodie.insert.shuffle.parallelism", 2)
	      .option("hoodie.upsert.shuffle.parallelism", 2)
	      .option("PRECOMBINE_FIELD_OPT_KEY", "ts") //指定提交时间列
	      .option("RECORDKEY_FIELD_OPT_KEY", "uuid") //指定uuid唯一标示列
	      .option("hoodie.table.name", "wzyTable")
	      .option("hoodie.datasource.write.partitionpath.field", "hudipart") //分区列
	      .mode(SaveMode.Overwrite)
	      .save("/wzy/data/hudi")
	  }
	
	  /**
	   * 更新数据
	   * @param sparkSession
	   * @return
	   */
	  def updateData(sparkSession: SparkSession) = {
	    import org.apache.spark.sql.functions._
	    import sparkSession.implicits._
	    val commitTime = System.currentTimeMillis().toString //生成提交时间
	    val df = sparkSession.read.text("/test/test22")
	      .mapPartitions(partitions => {
	        partitions.map(item => {
	          val jsonObject = JsonUtil.getJsonData(item.getString(0))
	          WzyEntity(jsonObject.getIntValue("uid"), jsonObject.getString("uname"), jsonObject.getString("dt"))
	        })
	      })
	    val result = df.withColumn("ts", lit(commitTime)) //添加ts 时间戳列
	      .withColumn("uuid", col("uid")) //添加uuid 列
	      .withColumn("hudipart", col("dt")) //增加hudi分区列
	    result.write.format("org.apache.hudi")
	      .option("hoodie.insert.shuffle.parallelism", 2)
	      .option("hoodie.upsert.shuffle.parallelism", 2)
	      .option("PRECOMBINE_FIELD_OPT_KEY", "ts") //指定提交时间列
	      .option("RECORDKEY_FIELD_OPT_KEY", "uuid") //指定uuid唯一标示列
	      .option("hoodie.table.name", "wzyTable")
	      .option("hoodie.datasource.write.partitionpath.field", "hudipart") //分区列
	      .mode(SaveMode.Append)
	      .save("/wzy/data/hudi")
	  }
	
	  /**
	   * 查询数据
	   * @param sparkSession
	   * @return
	   */
	
	  def queryData(sparkSession: SparkSession) = {
	    val df = sparkSession.read.format("org.apache.hudi")
	      .load("/wzy/data/hudi/*/*")
	    df.show()
	    println(df.count())
	  }
	
	  /**
	   * 设置起始时间查询
	   * @param sparkSession
	   * @return
	   */
	  def incrementalQuery(sparkSession: SparkSession) = {
	    val beginTime = 20210105072320l
	    val df = sparkSession.read.format("org.apache.hudi")
	      .option(DataSourceReadOptions.QUERY_TYPE_OPT_KEY, DataSourceReadOptions.QUERY_TYPE_INCREMENTAL_OPT_VAL) //指定模式为增量查询
	      .option(DataSourceReadOptions.BEGIN_INSTANTTIME_OPT_KEY, beginTime) //设置开始查询的时间戳  不需要设置结束时间戳
	      .load("/wzy/data/hudi")
	    df.show()
	    println(df.count())
	  }
	
	
	  /**
	   * 按照时间起始时间查询
	   * @param sparkSession
	   * @return
	   */
	  def  timeQuery(sparkSession: SparkSession) = {
	    val beginTime = 20210105072320l
	    val endTime = 20210105073142l
	    val df = sparkSession.read.format("org.apache.hudi")
	      .option(DataSourceReadOptions.QUERY_TYPE_OPT_KEY, DataSourceReadOptions.QUERY_TYPE_INCREMENTAL_OPT_VAL) //指定模式为增量查询
	      .option(DataSourceReadOptions.BEGIN_INSTANTTIME_OPT_KEY, beginTime) //设置开始查询的时间戳
	      .option(DataSourceReadOptions.END_INSTANTTIME_OPT_KEY, endTime)
	      .load("/wzy/data/hudi")
	    df.show()
	    println(df.count())
	  }
	
	}




## 更新

updte 和 insert的区别在于saveMode从overwrite变为append

### 造假数据

vim test22

	{'uid':1,'uname':'xiaobai1','dt':'2020/10'}
	{'uid':2,'uname':'xiaohong1','dt':'2020/11'}

放入Hadop

	hdfs dfs -put test22 /test/

## 查询

### queryData

查询结果


	+-------------------+--------------------+------------------+----------------------+--------------------+---+---------+-------+-------------+----+--------+
	|_hoodie_commit_time|_hoodie_commit_seqno|_hoodie_record_key|_hoodie_partition_path|   _hoodie_file_name|uid|    uname|     dt|           ts|uuid|hudipart|
	+-------------------+--------------------+------------------+----------------------+--------------------+---+---------+-------+-------------+----+--------+
	|     20210227212809|  20210227212809_0_1|                 2|               2020/11|dda06899-3c3b-48e...|  2|xiaohong1|2020/11|1614432485895|   2| 2020/11|
	|     20210227212542|  20210227212542_0_2|                 2|               2020/09|2f8149ae-73df-44f...|  2| xiaohong|2020/09|1614432339009|   2| 2020/09|
	|     20210227212809|  20210227212809_1_2|                 1|               2020/10|7598dfe5-d7f8-491...|  1| xiaobai1|2020/10|1614432485895|   1| 2020/10|
	|     20210227212542|  20210227212542_1_1|                 1|               2020/08|0f15f096-e883-44c...|  1|  xiaobai|2020/08|1614432339009|   1| 2020/08|
	+-------------------+--------------------+------------------+----------------------+--------------------+---+---------+-------+-------------+----+--------+




