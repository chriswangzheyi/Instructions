# SparkSQL自适应分区

## 概念

自适应分区是Spark 3中推出的很靠谱的功能。因为它可以动态得根据数据量来适配任务。而不是固定分区数量，导致在生产环境中一些分区空转。还有一个非常重要的解决倾斜的配置。默认是不开启，强烈建议打开。

## 操作步骤

	# 开启自适应分区
	spark.sql.adaptive.enabled  true
	# 开启自动处理倾斜
	spark.sql.adaptive.skewJoin.enabled true
	# 倾斜因子
	spark.sql.adaptive.skewJoin.skewedPartitionFactor 5
	

你会发现开启它会和不开启任务的执行会有很大的差别。你可能会惊恐，为什么很多的任务被skip掉了？

Spark SQL牛叉得可以动态地检测分区的数据量，超过阈值自动调度任务处理。在有倾斜的场景，性能数十倍、数百倍增长。


## 广播

广播优化是经常提及的。我们可以控制广播阈值来定义哪些表该广播、哪些不该广播。但需要考虑的是：广播怎么就知道表的大小呢？答案是：你需要做ANALYZE。不然，这将成为一个笑话。

	spark.sql.autoBroadcastJoinThreshold    # 默认10MB

千万不要把广播设得太大，不然，你会发现，跑大作业的时候，你的Web UI都会死掉。广播对Driver是有压力的。Driver出现问题，有人想过去换GC，其实，大部分这是徒劳的。即使你用G1。结果一样是一摊屎。

针对一些比较特殊的，我们可以强制让JOIN走SHUFFLE。所以，下面这些HINT，你得知道。

	SELECT /*+ COALESCE(3) */ * FROM t
	SELECT /*+ REPARTITION(3) */ * FROM t
	SELECT /*+ REPARTITION(c) */ * FROM t
	SELECT /*+ REPARTITION(3, c) */ * FROM t
	SELECT /*+ REPARTITION_BY_RANGE(c) */ * FROM t
	SELECT /*+ REPARTITION_BY_RANGE(3, c) */ * FROM t
	

## CBO

CBO是基于代价的优化，对执行计划的优化也是可观的。它会根据数据的大小、分布以及算子的特点选择出来比较好的物理执行计划。打开CBO，一些大的作业的执行计划也可以看到明显的变化。

	spark.sql.cbo.enabled   true
	spark.sql.cbo.joinReorder.enabled   true
	spark.sql.cbo.planStats.enabled true
	
## Resource Dynamic Allocation

集群的资源是宝贵的，买服务器、建机房的时候就知道了。所以，我们不要上来就分配特别大的资源。当跑完一些作业后，应该把资源让给其他应用。配合FAIR Shceulder，可以将资源尽可能地复用。

	spark.dynamicAllocation.enabled true
	
注意哦！一个定在生产上把yarn spark shuffle service配置好。否则，你会发现，Container会不停地释放、申请、释放、申请。任务根本执行不下去。
