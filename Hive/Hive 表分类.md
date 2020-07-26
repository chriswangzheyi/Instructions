# Hive 表分类

## 托管表（内部表）

hive 默认创建的表都是托管表，hive控制其数据的生命周期。删除托管表时，元数据和数据都被删除。


	$>hive>create table if not exists myhive.employee_inner(eid int, name String, salary String, destination String) COMMENT 'Employee details' ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n' STORED AS TEXTFILE;

	$>hive> drop table myhive.employee_inner;



## 外部表 

hive控制元数据，删除托管表时，删除元数据，数据不被删除

	$>hive>create external table if not exists myhive.employee_outer(eid int, name String, salary String, destination String) COMMENT 'Employee details' ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n' STORED AS TEXTFILE;


跟内部表的区别就是创建表的时候使用了 external_table

## 分区表

避免单个节点的数据量过大


	$>hive>create table if not exists myhive.test2(eid int, name String, salary String, destination String) partitioned by(country string,state string) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n' STORED AS TEXTFILE;

使用了 partitioned by 作为分区依据。上面的例子是两级分区。

### 加载数据到指定分区
		$>hive>load data local inpath '/home/hadoop/Desktop/employee.txt' into table myhive.test2 partition(country='china',state='shanxi');



