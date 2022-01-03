# Spark 和 MR 的区别

##排序问题：

MR是全局排序，Spark是基于Hash排序。功能上，MR的shuffle和Spark的shuffle都是对Map端的数据进行分区。

##容错：

MR出错需要从头执行job，Spark利用DAG只需从出错的那一步开始

##Shuffle：

MR的task会产生多个小文件，且无法复用，Spark 会合并小文件

##交换数据方式：

MR使用hdfs做数据交换，Spark用本地磁盘+内存
