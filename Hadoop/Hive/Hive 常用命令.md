# Hive 常用命令

参考资料：https://www.jianshu.com/p/ef429de8a126

## 数据库
 查询数据库列表

	show databases ;

 使用指定的数据库
 
	use default;

 查看数据库的描述信息
 
	desc database extended db_hive_03 ;

## 表

查询表列表

	show tables ;

查询表的描述信息:

	desc student ;
	desc extended student ;
	desc formatted student ;

创建表

	vi /root/test.txt

	1,aa
	2,bb
	3,cc

Hive中：

	create table student(
	id int, 
	name string) 
	ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';
	load data local inpath '/root/test.txt'into table student ;



建表的时候分隔符是逗号


创建一张表并复制一个表的结构和数据

	create table if not exists default.dept_cats as select * from dept ;


 使用另一张表的结构创建一张新表

	create table if not exists default.dept_like like default.dept ;

清空表：

	truncate table dept_cats ;

删除表

	drop table if exists dept_like_rename ;

修改表名

	alter table dept_like rename to dept_like_rename ;

查询表

	select * from student ;
	select id from student ;


## 功能函数:

**显示功能函数列表**

	show functions ;

查看功能函数的描述信息

	desc function upper ;

查询功能函数的扩展信息

	desc function extended upper ;

测试功能函数

	select id ,upper(name) uname from db_hive.student ;


## 进阶

**创建一个外部表，并指定导入文件的位置和字段分割符：**

	create EXTERNAL table IF NOT EXISTS default.emp_ext2(
	empno int,
	ename string,
	job string,
	mgr int,
	hiredate string,
	sal double,
	comm double,
	deptno int
	)
	ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
	location '/user/hive/warehouse/emp_ext2';

** 创建分区表：**

	create EXTERNAL table IF NOT EXISTS default.emp_partition(
	empno int,
	ename string,
	job string,
	mgr int,
	hiredate string,
	sal double,
	comm double,
	deptno int
	)
	partitioned by (month string,day string)
	ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' ;

向分区表中导入数据：

	load data local inpath '/usr/local/app/hive_test/emp.txt' into table default.emp_partition partition (month='201805',day='31') ;


查看分区表列表：

	show partitions emp_partition ;


查询分区表中的数据：

	select * from emp_partition where month = '201509' and day = '13' ;

加载数据到hive：

	1）加载本地文件到hive表
	
	load data local inpath '/opt/datas/emp.txt' into table default.emp ;
	
	2）加载hdfs文件到hive中
	load data inpath '/user/beifeng/hive/datas/emp.txt' overwrite into table default.emp ;
	
	3）加载数据覆盖表中已有的数据
	load data inpath '/user/beifeng/hive/datas/emp.txt' into table default.emp ;
	
	4）创建表是通过insert加载
	create table default.emp_ci like emp ;
	insert into table default.emp_ci select * from default.emp ;
	
	5）创建表的时候通过location指定加载
	7. hive到文件：
	insert overwrite local directory '/opt/datas/hive_exp_emp'
	select * from default.emp ;
	
	insert overwrite local directory '/opt/datas/hive_exp_emp2'
	ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' COLLECTION ITEMS TERMINATED BY '\n'
	select * from default.emp ;
	
	bin/hive -e "select * from default.emp ;" > /opt/datas/exp_res.txt


将查询结果导出到本地文件中：

	insert overwrite directory '/hive_test/export_emp.txt' select * from emp;

	select * from emp ;

	select t.empno, t.ename, t.deptno from emp t ;