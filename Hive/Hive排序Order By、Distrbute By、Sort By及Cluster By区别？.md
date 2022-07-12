# Hive排序Order By、Distrbute By、Sort By及Cluster By区别？

1）Order By：全局排序，只有一个Reducer；Total Job=1,可以在运行日志中看到number of reucers=1.用在select语句的后面。

2）Distrbute By：分区排序, 类似MR中Partition，进行分区，结合sort by使用;

3）Sort By：分区内有序；每个reducer内部进行排序，对全局结果集来说不是排序。随机分区，防止数据倾斜。①设置reduce个数。set mapreduce.job.reducers=3;②查看reduce个数。set mapreduce.job.reducers;

4）Cluster By：当Distribute by和Sorts by字段相同时，可以使用Cluster by方式。Cluster by除了具有Distribute by的功能外还兼具Sort by的功能。但是排序只能是升序排序，不能指定排序规则为ASC或者DESC
