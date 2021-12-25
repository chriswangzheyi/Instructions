# Spark 和 MR 的区别

排序问题：
MR是全局排序，Spark是基于Hash排序。功能上，MR的shuffle和Spark的shuffle都是对Map端的数据进行分区。