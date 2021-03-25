#13 Hudi整合Hive

## 将hudi的包拷贝到hive中

	cp hudi-hadoop-mr-bundle-0.6.0-SNAPSHOT.jar /usr/local/hive/lib/


## 测试数据

	[root@master ~]# hdfs dfs -cat /test/test11
	{'uid':1,'uname':'xiaobai','dt':'2020/08'}
	{'uid':2,'uname':'xiaohong','dt':'2020/09'}

	[root@master ~]# hdfs dfs -cat /test/test22
	{'uid':1,'uname':'xiaobai1','dt':'2020/10'}
	{'uid':2,'uname':'xiaohong1','dt':'2020/11'}


## 查询

### 运行后

#### hdfs

	[root@master hadoop]# hdfs dfs -ls /wzy/data_hive/hudi1/2020
	Found 2 items
	drwxr-xr-x   - root supergroup          0 2021-03-07 10:54 /wzy/data_hive/hudi1/2020/08
	drwxr-xr-x   - root supergroup          0 2021-03-07 10:54 /wzy/data_hive/hudi1/2020/09

#### hive

	hive> use default;
	OK
	hive> select * from wzyhivetable1;
	20210307235341	20210307235341_1_2	1	2020/08	0f13d0be-85cb-4b9b-90ff-a40045b57ac8-0_1-21-23_20210307235341.parquet	1	xiaobai	1615132417306	1	2020/08	2020	08
	20210307235341	20210307235341_0_1	2	2020/09	8cb912a8-6cdf-456e-adbb-7a6d777bf288-0_0-21-22_20210307235341.parquet	2	xiaohong	1615132417306	2	2020/09	2020	09




## 更新

### 运行后

#### hdfs

	[root@master ~]# hdfs dfs -ls /wzy/data_hive/hudi1/2020
	Found 4 items


	drwxr-xr-x   - root supergroup          0 2021-03-08 08:28 /wzy/data_hive/hudi1/2020/08
	drwxr-xr-x   - root supergroup          0 2021-03-08 08:28 /wzy/data_hive/hudi1/2020/09
	drwxr-xr-x   - root supergroup          0 2021-03-08 08:29 /wzy/data_hive/hudi1/2020/10
	drwxr-xr-x   - root supergroup          0 2021-03-08 08:29 /wzy/data_hive/hudi1/2020/11


#### hive

	20210308212805	20210308212805_1_2	1	2020/10	410db936-e58d-48a0-93dc-79ba290c88ba-0_1-21-31_20210308212805.parquet	1	xiaobai1	1615210081870	1	2020/10	2020	10
	20210308212805	20210308212805_0_1	2	2020/11	b9d49505-fdf6-4bb4-8fda-423f8c472c39-0_0-21-30_20210308212805.parquet	2	xiaohong1	1615210081870	2	2020/11	2020	11
	Time taken: 1.582 seconds, Fetched: 2 row(s)

## 代码

	package com.wzy.test
	
	import com.wzy.entity.WzyEntity
	import com.wzy.util.JsonUtil
	import org.apache.hudi.config.HoodieIndexConfig
	import org.apache.hudi.hive.MultiPartKeysValueExtractor
	import org.apache.hudi.index.HoodieIndex
	import org.apache.hudi.{DataSourceReadOptions, DataSourceWriteOptions}
	import org.apache.spark.SparkConf
	import org.apache.spark.sql.{SaveMode, SparkSession}
	
	object HudiHiveTest {
	
	  def main(args: Array[String]): Unit = {
	    System.setProperty("HADOOP_USER_NAME", "root")
	    val sparkConf = new SparkConf().setAppName("HudiTest")
	      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
	      .setMaster("local[*]")
	    val sparkSession = SparkSession.builder().config(sparkConf).enableHiveSupport().getOrCreate()
	    val ssc = sparkSession.sparkContext
	
	    ssc.hadoopConfiguration.set("dfs.client.use.datanode.hostname","true");
	    //insertData(sparkSession)
	    updateData(sparkSession)
	    //queryData(sparkSession)
	    //incrementalQuery(sparkSession)
	    //timeQuery(sparkSession)
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
	
	    //使用cow_TABLE_TPYE_OPT_VAL的方式存储data
	    result.write.format("org.apache.hudi")
	      .option("hoodie.insert.shuffle.parallelism", 2)
	      .option("hoodie.upsert.shuffle.parallelism", 2)
	      .option("PRECOMBINE_FIELD_OPT_KEY", "ts") //指定提交时间列
	      .option("RECORDKEY_FIELD_OPT_KEY", "uuid") //指定uuid唯一标示列
	      .option("hoodie.table.name", "wzyHiveTable1")
	      .option("hoodie.datasource.write.partitionpath.field", "hudipart") //分区列
	      .option(DataSourceWriteOptions.TABLE_TYPE_OPT_KEY, DataSourceWriteOptions.MOR_TABLE_TYPE_OPT_VAL)
	      .option(DataSourceWriteOptions.HIVE_URL_OPT_KEY, "jdbc:hive2://192.168.195.150:10000/default") //hive地址
	      .option(DataSourceWriteOptions.HIVE_DATABASE_OPT_KEY, "default") //设置hudi与hive同步的数据库
	      .option(DataSourceWriteOptions.HIVE_TABLE_OPT_KEY, "wzyHiveTable1") //设置hudi与hive同步的表名
	      .option(DataSourceWriteOptions.HIVE_PARTITION_FIELDS_OPT_KEY, "dt,ds") //hive表同步的分区列
	      .option(DataSourceWriteOptions.HIVE_PARTITION_EXTRACTOR_CLASS_OPT_KEY, classOf[MultiPartKeysValueExtractor].getName) // 分区提取器 按/ 提取分区
	      .option(DataSourceWriteOptions.HIVE_SYNC_ENABLED_OPT_KEY, "true") //设置数据集注册并同步到hive
	      .option(HoodieIndexConfig.BLOOM_INDEX_UPDATE_PARTITION_PATH, "true") //设置当分区变更时，当前数据的分区目录是否变更
	      .option(HoodieIndexConfig.INDEX_TYPE_PROP, HoodieIndex.IndexType.GLOBAL_BLOOM.name()) //设置索引类型目前有HBASE,INMEMORY,BLOOM,GLOBAL_BLOOM 四种索引 为了保证分区变更后能找到必须设置全局GLOBAL_BLOOM
	      .mode(SaveMode.Overwrite)
	      .save("/wzy/data_hive/hudi1")
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
	      .option("hoodie.table.name", "wzyHiveTable1")
	      .option("hoodie.datasource.write.partitionpath.field", "hudipart") //分区列
	      .option(DataSourceWriteOptions.TABLE_TYPE_OPT_KEY, DataSourceWriteOptions.MOR_TABLE_TYPE_OPT_VAL)
	      .option(DataSourceWriteOptions.HIVE_URL_OPT_KEY, "jdbc:hive2://192.168.195.150:10000/default") //hive地址
	      .option(DataSourceWriteOptions.HIVE_DATABASE_OPT_KEY, "default") //设置hudi与hive同步的数据库
	      .option(DataSourceWriteOptions.HIVE_TABLE_OPT_KEY, "wzyHiveTable1") //设置hudi与hive同步的表名
	      .option(DataSourceWriteOptions.HIVE_PARTITION_FIELDS_OPT_KEY, "dt,ds") //hive表同步的分区列
	      .option(DataSourceWriteOptions.HIVE_PARTITION_EXTRACTOR_CLASS_OPT_KEY, classOf[MultiPartKeysValueExtractor].getName) // 分区提取器 按/ 提取分区
	      .option(DataSourceWriteOptions.HIVE_SYNC_ENABLED_OPT_KEY, "true") //设置数据集注册并同步到hive
	      .option(HoodieIndexConfig.BLOOM_INDEX_UPDATE_PARTITION_PATH, "true") //设置当分区变更时，当前数据的分区目录是否变更
	      .option(HoodieIndexConfig.INDEX_TYPE_PROP, HoodieIndex.IndexType.GLOBAL_BLOOM.name()) //设置索引类型目前有HBASE,INMEMORY,BLOOM,GLOBAL_BLOOM 四种索引 为了保证分区变更后能找到必须设置全局GLOBAL_BLOOM
	      .mode(SaveMode.Append)
	      .save("/wzy/data_hive/hudi1")
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
