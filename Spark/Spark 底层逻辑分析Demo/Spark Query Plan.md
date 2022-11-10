# Spark Query Plan

参考：https://www.youtube.com/watch?v=UZt_tqx4sII&list=PLmtsMNDRU0Bw6VnJ2iixEwxmOZNT7GDoC&index=1

## 前置条件

安装好Spark

## 操作步骤

### 进入脚本

	spark-shell --master local

### Demo1

	val simpleNumbers = spark.range(1, 1000000)
	val times5 = simpleNumbers.selectExpr(" id * 5 as id")
	times5.show()
	
解释命令：

	times5.explain()
	
打印：

	== Physical Plan ==
	*(1) Project [(id#76L * 5) AS id#78L]
	+- *(1) Range (1, 1000000, step=1, splits=1)

解释：

* splits=1 表示1个partition
	

### Demo2

	val moreNumbers = spark.range(1, 1000000, 2)
	val split7 = moreNumbers.repartition(7)
	split7.explain()

打印：

	== Physical Plan ==
	Exchange RoundRobinPartitioning(7), REPARTITION_WITH_NUM, [id=#271]
	+- *(1) Range (1, 1000000, step=2, splits=1)

解释：

* Exchange 表示Shuffle
* RoundRobinPartitioning(7) 表示以RoundRobin的方式分配到7个Partition中

### Demo3

	val ds1 = spark.range(1,10000000)
	val ds2 = spark.range(1,10000000,2)
	val ds3 = ds1.repartition(7)
	val ds4 = ds2.repartition(9)
	val ds5 = ds3.selectExpr(" id * 5 as id ")
	val joined = ds5.join(ds4, "id")
	val sum = joined.selectExpr("sum(id)")
	sum.explain()

打印：

	== Physical Plan ==
	*(7) HashAggregate(keys=[], functions=[sum(id#113L)])
	+- Exchange SinglePartition, ENSURE_REQUIREMENTS, [id=#402]
	   +- *(6) HashAggregate(keys=[], functions=[partial_sum(id#113L)])
	      +- *(6) Project [id#113L]
	         +- *(6) SortMergeJoin [id#113L], [id#107L], Inner
	            :- *(3) Sort [id#113L ASC NULLS FIRST], false, 0
	            :  +- Exchange hashpartitioning(id#113L, 200), ENSURE_REQUIREMENTS, [id=#386]
	            :     +- *(2) Project [(id#105L * 5) AS id#113L]
	            :        +- Exchange RoundRobinPartitioning(7), REPARTITION_WITH_NUM, [id=#382]
	            :           +- *(1) Range (1, 10000000, step=1, splits=1)
	            +- *(5) Sort [id#107L ASC NULLS FIRST], false, 0
	               +- Exchange hashpartitioning(id#107L, 200), ENSURE_REQUIREMENTS, [id=#393]
	                  +- Exchange RoundRobinPartitioning(9), REPARTITION_WITH_NUM, [id=#392]
	                     +- *(4) Range (1, 10000000, step=2, splits=1)
	                  
解释：

读法：从stage1到4，反着读。

* +- *(3) ： 数字表示Stage
* +-  +- Exchange RoundRobinPartitioning(7), REPARTITION_WITH_NUM, [id=#382]  ： 表示在Stage1 和 stage3直接的操作
* id#97L ： 表示字段和字段id
* HashAggregate: 求和


