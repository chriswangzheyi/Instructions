# Hive语法

参考：https://www.cnblogs.com/braveym/p/6622336.html

## 基本语法

### 创建数据库

	create database name;

## DDL(Data Defination Language)

### 新建表

	CREATE [EXTERNAL] TABLE [IF NOT EXISTS] table_name
	
	[(col_name data_type [COMMENT col_comment], ...)]
	
	[COMMENT table_comment]
	
	[PARTITIONED BY (col_name data_type [COMMENT col_comment], ...)]
	
	[CLUSTERED BY (col_name, col_name, ...)
	
	[SORTED BY (col_name [ASC|DESC], ...)] INTO num_buckets BUCKETS]
	
	[ROW FORMAT row_format]
	
	[STORED AS file_format]
	
	[LOCATION hdfs_path] 


例子：

	CREATE TABLE person(name STRING,age INT);


### 删除表

	DROP TABLE table_name;

例子：

	DROP TABLE person;


### 更新表


#### 增加字段



#### 


### 查看表结构

	DESC table_name;
	
例子：

	DESC person

### 