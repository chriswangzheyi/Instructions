# Hive与Impala 异同


##不同点总

* Hive适合于长时间的批处理查询分析，而Impala适合于实时交互式SQL查询。
* Hive依赖于MapReduce计算框架，Impala把执行计划表现为一棵完整的执行计划树，直接分发执行计划到各个Impalad执行查询。
* Hive在执行过程中，如果内存放不下所有数据，则会使用外存，以保证查询能顺序执行完成，而Impala在遇到内存放不下数据时，不会利用外存，所以Impala目前处理查询时会受到一定的限制。

##相同点

* Hive与Impala使用相同的存储数据池，都支持把数据存储于HDFS和HBase中。
* Hive与Impala使用相同的元数据。
* Hive与Impala中对SQL的解释处理比较相似，都是通过词法分析生成执行计划。

##总结：

* Impala的目的不在于替换现有的MapReduce工具。
* 把Hive与Impala配合使用效果最佳。
* 可以先使用Hive进行数据转换处理，之后再使用Impala在Hive处理后的结果数据集上进行快速的数据分析