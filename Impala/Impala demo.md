# Impala demo


## 启动

在Impala Daemon所在节点输入

	impala-shell
	
	
##  impala 查询处理

### database

####创建数据库：

	-- 示例：
	CREATE DATABASE IF NOT EXISTS database_name;

#### 删除数据库：

	-- 语法：
	DROP (DATABASE|SCHEMA) [IF EXISTS] database_name [RESTRICT | 
	CASCADE] [LOCATION hdfs_path];

	-- 示例：
	DROP DATABASE IF EXISTS sample_database;

####选择数据库：

	-- 语法：
	USE db_name;
	

### table

#### 创建表：

	-- 语法：
	create table IF NOT EXISTS database_name.table_name (
	   column1 data_type,
	   column2 data_type,
	   column3 data_type,
	   ………
	   columnN data_type
	);

	-- 示例：
	CREATE TABLE IF NOT EXISTS my_db.student
	   (name STRING, age INT, contact INT );

####插入表：

	-- 语法：
	insert into table_name (column1, column2, column3,...columnN) values (value1, value2, value3,...valueN);
	insert overwrite table_name values (value1, value2, value2);

	-- 示例：
	insert into employee (ID,NAME,AGE,ADDRESS,SALARY)VALUES (1, 'Ramesh', 32, 'Ahmedabad', 20000 );
	insert overwrite employee values (1, 'Ram', 26, 'Vishakhapatnam', 37000 );

####查询表：

	-- 语法：
	SELECT column1, column2, columnN from table_name;

	--示例：
	select name, age from customers; 

####表描述：

	-- 语法：
	describe table_name;

	-- 示例：
	describe customer;
	
####修改表（重命名表案例，其它自行查阅）：

	-- 语法：
	ALTER TABLE [old_db_name.]old_table_name RENAME TO [new_db_name.]new_table_name

	-- 示例：
	ALTER TABLE my_db.customers RENAME TO my_db.users;
	
####删除表：

	-- 语法：
	DROP table database_name.table_name;
	
	--示例：
	drop table if exists my_db.student;

####截断表：

	-- 语法：
	truncate table_name;
	
	-- 示例：
	truncate customers;

####显示表：

	show tables 

####创建视图：

	-- 语法：
	Create View IF NOT EXISTS view_name as Select statement
	-- 示例：
	CREATE VIEW IF NOT EXISTS customers_view AS select name, age from customers;

####修改视图：

	-- 语法
	ALTER VIEW database_name.view_name为Select语句
	-- 示例
	Alter view customers_view as select id, name, salary from customers;

####删除视图：

	-- 语法：
	DROP VIEW database_name.view_name;
	-- 示例：
	Drop view customers_view;

### 条件

####order by 子句：

	--语法
	select * from table_name ORDER BY col_name [ASC|DESC] [NULLS FIRST|NULLS LAST]
	--示例
	Select * from customers ORDER BY id asc;

####group by 字句：

	-- 语法
	select data from table_name Group BY col_name;
	-- 示例
	Select name, sum(salary) from customers Group BY name;

####having 子句：

	--语法
	select * from table_name ORDER BY col_name [ASC|DESC] [NULLS FIRST|NULLS LAST]
	-- 示例
	select max(salary) from customers group by age having max(salary) > 20000;

####limit限制：

	-- 语法：
	select * from table_name order by id limit numerical_expression;

####offset偏移：

	-- 示例：
	select * from customers order by id limit 4 offset 0;

####union聚合：

	-- 语法：
	query1 union query2;
	-- 示例：
	select * from customers order by id limit 3
	union select * from employee order by id limit 3;

####with子句

	-- 语法：
	with x as (select 1), y as (select 2) (select * from x union y);
	-- 示例：
	with t1 as (select * from customers where age>25), 
	   t2 as (select * from employee where age>25) 
	   (select * from t1 union select * from t2);

####distinct去重：

	-- 语法：
	select distinct columns… from table_name;
	-- 示例：
	select distinct id, name, age, salary from customers; 