# Hive 分桶

## 概念

在分区数量过于庞大以至于可能导致文件系统崩溃时，我们就需要使用分桶来解决问题

分桶是相对分区进行更细粒度的划分。分桶则是指定分桶表的某一列，让该列数据按照哈希取模的方式随机、均匀的分发到各个桶文件中。因为分桶操作需要根据某一列具体数据来进行哈希取模操作，故指定的分桶列必须基于表中的某一列(字段) 要使用关键字clustered by 指定分区依据的列名，还要指定分为多少桶

	create table test(id int,name string) cluster by (id) into 5 buckets .......
	
	insert into buck select id ,name from p cluster by (id)

## Hive分区分桶区别

* 分区是表的部分列的集合，可以为频繁使用的数据建立分区，这样查找分区中的数据时就不需要扫描全表，这对于提高查找效率很有帮助
* 不同于分区对列直接进行拆分，桶往往使用列的哈希值对数据打散，并分发到各个不同的桶中从而完成数据的分桶过程
* 分区和分桶最大的区别就是分桶随机分割数据库，分区是非随机分割数据库


### 举例： 

分区表 按 天/月/年

分桶表 按 用户id