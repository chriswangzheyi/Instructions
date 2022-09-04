# spark sql 非业务调优

https://www.yisu.com/zixun/573651.html

##jvm调优

GC及数据结构调优

## 内存调优

spark2.+采用：

	spark.catalog.cacheTable("tableName")缓存表，spark.catalog.uncacheTable("tableName")解除缓存。
spark 1.+采用：

	sqlContext.cacheTable("tableName")缓存，sqlContext.uncacheTable("tableName") 解除缓存。
	
##广播

大小表进行join时，广播小表到所有的Worker节点，来提升性能是一个不错的选择。

	spark.sql.broadcastTimeout 广播等待超时时间，单位秒
	spark.sql.autoBroadcastJoinThreshold 最大广播表的大小
	
## 分区数据的调控

分区设置spark.sql.shuffle.partitions，默认是200.

对于有些公司来说，估计在用的时候会有Spark sql处理的数据比较少，然后资源也比较少，这时候这个shuffle分区数200就太大了，应该适当调小，来提升性能。

也有一些公司，估计在处理离线数据，数据量特别大，而且资源足，这时候shuffle分区数200，明显不够了，要适当调大。

## 文件与分区

一个是在读取文件的时候一个分区接受多少数据；

另一个是文件打开的开销，通俗理解就是小文件合并的阈值。

	spark.sql.files.maxPartitionBytes 打包传入一个分区的最大字节，在读取文件的时候。
	spark.sql.files.openCostInBytes 用相同时间内可以扫描的数据的大小来衡量打开一个文件的开销。当将多个文件写入同一个分区的时候该参数有用。该值设置大一点有好处，有小文件的分区会比大文件分区处理速度更快（优先调度）。
	
## 文件格式

建议parquet或者orc。Parquet已经可以达到很大的性能了