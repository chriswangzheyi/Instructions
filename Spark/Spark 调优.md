# Spark 调优

参考：https://zhuanlan.zhihu.com/p/54293797

1. 资源调优
1. 并行度调优
1. 代码调优
1. 数据本地化
1. 内存调优
1. shuffle 参数
1. 堆外内存
1. 数据倾斜


## 资源调优

* SPARK_WORK_CORES 指定每个 worker 分配的核数
* SPARK_WORK_MEMORY 指定每个 worker 分配的内存
* SPARK_WORK_INSTANCES 指定每台机器启动的 worker 数量
* spark.dynamicAllocation.enabled true 开启动态资源分配


## 并行度优化


## 代码调优

避免创建重复的 RDD, 可以复用同一个 RDD
对多次使用的 RDD 进行持久化
持久化性能最优的是 MEMORY_ONLY ，但是前提是我们的内存必须足够大，要不然很容易导致 OOM


## 内存调优

Spark JVM 调优主要是降低 GC 的时间

### spark中executor内存划分

Executor的内存主要分为三块：

* 第一块是让task执行我们自己编写的代码时使用；
* 第二块是让task通过shuffle过程拉取了上一个stage的task的输出后，进行聚合等操作时使用；
* 第三块是让RDD缓存时使用。

实际上就是把我们的一个executor分成了三部分：

* 一部分是Storage内存区域，默认0.6；
* 一部分是execution区域，默认0.2；
* 还有一部分是其他区域，默认0.2。

用这几个参数去控制：

* spark.storage.memoryFraction：默认0.6（cache数据大，则应调高）
* spark.shuffle.memoryFraction：默认0.2  （shuffle多，则应调高）


## 使用高性能算子

（1）使用reduceByKey/aggregateByKey替代groupByKey；

（2）使用mapPartitions替代普通map；mapPartitions一次函数调用会处理一个partition所有的数据。

（3）使用foreachPartitions替代foreach；一次函数调用处理一个partition的所有数据。

（4）使用filter之后进行coalesce操作；

* 使用coalesce算子，手动减少RDD的partition数量；

（5）使用repartitionAndSortWithinPartitions替代repartition与sort类操作；

* 如果需要在repartition重分区之后，还要进行排序，建议直接使用repartitionAndSortWithinPartitions算子。
* 因为该算子可以一边进行重分区的shuffle操作，一边进行排序。

## 尽量避免使用shuffle类算子

（1）shuffle过程，就是将分布在集群中多个节点上的同一个key，拉取到同一个节点上，进行聚合或join等操作。

（2）shuffle涉及到数据要进行大量的网络传输。

（3）使用reduceByKey、join、distinct、repartition等算子操作，这里都会产生shuffle。

（4）避免产生shuffle：Broadcast+map的join操作，不会导致shuffle操作。

（5）使用map-side预聚合的shuffle操作，减少数据的传输量，提升性能。

* map-side预聚合，在每个节点本地对相同的key进行一次聚合操作，类似于MapReduce中的本地combiner。
* 建议使用reduceByKey或者aggregateByKey算子来替代掉groupByKey算子。因为reduceByKey和aggregateByKey算子都会使用用户自定义的函数对每个节点本地的相同key进行预聚合。


