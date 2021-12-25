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
