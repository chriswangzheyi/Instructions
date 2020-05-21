# Hive语法

参考：https://www.cnblogs.com/braveym/p/6622336.html

https://www.jianshu.com/p/6383611b308d

## 基本语法

### 创建数据库

	create database name;

例子：

	create databases if not exists db_hive;
	create datebases if not exists db_hive localtion `db_hive2.db`

### 修改数据库

	alter database db_hive set dbproperties(`createtime`=`20170101`)
	//用户可以用alter database命令为数据库设置dbrpoprities属性值
	//此为属性信息，数据库的表名和数据库所在目录位置不可以修改


### 显示数据库

	show databases;//显示数据库
	show databases like 'db_hive*';//过滤显示查询数据库
	desc database db_hive;//显示数据库信息
	desc database extended db_hive;//显示数据库详细信息 
	use database db_hive;//使用数据库

### 删除数据库

	drop database db_hive2;//删除空数据库
	drop database if exists db_hive2;//判断数据库是否存在
	drop database db_hive2 cascade;//数据库不为空 用cascade强制删除 

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

	create table t1(
	    id      int
	   ,name    string
	   ,hobby   array<string>
	   ,add     map<String,string>
	)
	row format delimited
	fields terminated by ','
	collection items terminated by '-'
	map keys terminated by ':'
	;


例子2：

	//创建普通表
	create table if not exists student2(
	id int,name string
	)
	row format delimited fields terminated by '\t'
	stored as textfile
	location '/user/hive/warehouse/student2';
	
	//未被external修饰的是内部表（managed table），被external修饰的为外部表（external table）删除外部表的话只会删除元数据，数据不会被删除； 
	create external table if not exists default.dept(
	deptno int,
	dname string,
	loc int
	)
	row format delimited fields terminated by '\t';


### 分区表

	创建分区表
	hive (default)> create table dept_partition(
	               deptno int, dname string, loc string
	               )
	               partitioned by (month string)
	               row format delimited fields terminated by '\t';
	
	加载数据到分区表
	hive (default)> load data local inpath '/opt/module/datas/dept.txt' into table default.dept_partition partition(month='201709');
	hive (default)> load data local inpath '/opt/module/datas/dept.txt' into table default.dept_partition partition(month='201708');
	hive (default)> load data local inpath '/opt/module/datas/dept.txt' into table default.dept_partition partition(month='201707');
	
	查询分区表数据
	hive (default)> select * from dept_partition where month='201709'
	              union
	              select * from dept_partition where month='201708'
	              union
	              select * from dept_partition where month='201707';
	
	增加分区
	hive (default)>  alter table dept_partition add partition(month='201705') partition(month='201704');
	
	删除分区
	hive (default)> alter table dept_partition drop partition (month='201705'), partition (month='201706');
	
	查看分区表有多少分区
	hive>show partitions dept_partition;
	
	查看分区表结构
	hive>desc formatted dept_partition;
	
	# Partition Information          
	# col_name              data_type               comment             
	month                   string
	    
	//加载数据到分区
	hive (default)> load data local inpath '/opt/module/datas/dept.txt' into table default.dept_partition2 partition(month='201709', day='13');
	hive (default)> select * from dept_partition2 where month='201709' and day='13';
	
	//把数据直接上传到分区目录上，让分区表和数据产生关联的三种方式
	方式一：上传数据后修复
	hive (default)> dfs -mkdir -p /user/hive/warehouse/dept_partition2/month=201709/day=12;
	hive (default)> dfs -put /opt/module/datas/dept.txt  /user/hive/warehouse/dept_partition2/month=201709/day=12;
	hive (default)> msck repair table dept_partition2;
	hive (default)> select * from dept_partition2 where month='201709' and day='12';
	方式二：上传数据后添加分区
	hive (default)> dfs -mkdir -p /user/hive/warehouse/dept_partition2/month=201709/day=11;
	hive (default)> dfs -put /opt/module/datas/dept.txt  /user/hive/warehouse/dept_partition2/month=201709/day=11;
	hive (default)> alter table dept_partition2 add partition(month='201709', day='11');
	hive (default)> select * from dept_partition2 where month='201709' and day='11';
	方式三：上传数据后load数据到分区
	hive (default)> dfs -mkdir -p /user/hive/warehouse/dept_partition2/month=201709/day=10;
	hive (default)> load data local inpath '/opt/module/datas/dept.txt' into table dept_partition2 partition(month='201709',day='10');
	hive (default)> select * from dept_partition2 where month='201709' and day='10';
	
	修改表
	//重命名表名
	hive (default)> alter table dept_partition2 rename to dept_partition3;
	//添加列
	hive (default)> alter table dept_partition add columns(deptdesc string);
	//更新列
	hive (default)> alter table dept_partition change column deptdesc desc int;
	//替换列
	hive (default)> alter table dept_partition replace columns(deptno string, dname string, loc string);
	//删除表
	hive (default)> drop table dept_partition;
	/


### 装载数据


	LOAD DATA [LOCAL] INPATH 'filepath' [OVERWRITE] INTO TABLE tablename [PARTITION (partcol1=val1, partcol2=val2 ...)]


例子：

样本数据

	1,xiaoming,book-TV-code,beijing:chaoyang-shagnhai:pudong
	2,lilei,book-code,nanjing:jiangning-taiwan:taibei
	3,lihua,music-book,heilongjiang:haerbin

加载：

	load data local inpath '/root/data/1.data' overwrite into table t1;


验证:

	select * from t1;


### 导入导出数据

	数据导入
	//向表中装载数据（Load）
	hive>load data [local] inpath '/opt/module/datas/student.txt' [overwrite] into table student [partition (partcol1=val1,…)];
	//（1）load data:表示加载数据
	//（2）local:表示从本地加载数据到hive表；否则从HDFS加载数据到hive表
	//（3）inpath:表示加载数据的路径
	//（4）overwrite:表示覆盖表中已有数据，否则表示追加
	//（5）into table:表示加载到哪张表
	//（6）student:表示具体的表
	//（7）partition:表示上传到指定分区
	
	//通过查询语句向表中插入数据（Insert）
	hive (default)> from student
	              insert overwrite table student partition(month='201707')
	              select id, name where month='201709'
	              insert overwrite table student partition(month='201706')
	              select id, name where month='201709';
	
	// 查询语句中创建表并加载数据（As Select）
	//根据查询结果创建表（查询的结果会添加到新创建的表中）
	create table if not exists student3
	as select id, name from student;
	
	//创建表时通过Location指定加载数据路径
	hive (default)> create table if not exists student5(
	              id int, name string
	              )
	              row format delimited fields terminated by '\t'
	              location '/user/hive/warehouse/student5';
	hive (default)> dfs -put /opt/module/datas/student.txt  /user/hive/warehouse/student5;
	hive (default)> select * from student5;
	
	//Import数据到指定Hive表中
	hive (default)> import table student2 partition(month='201709') from '/user/hive/warehouse/export/student';
	
	数据导出
	//insert导入
	hive (default)> insert overwrite local directory '/opt/module/datas/export/student'  select * from student;
	//将查询结果格式化导入
	hive (default)> insert overwrite local directory '/opt/module/datas/export/student1'
	             ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
	             select * from student;
	//将查询的结果导出到HDFS上(没有local)
	hive (default)> insert overwrite directory '/user/atguigu/student2'
	             ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
	             select * from student;
	//Hadoop命令导出到本地
	hive (default)> dfs -get /user/hive/warehouse/student/month=201709/000000_0  /opt/module/datas/export/student3.txt;
	//Hive Shell 命令导出
	[atguigu@hadoop102 hive]$ bin/hive -e 'select * from default.student;' > /opt/module/datas/export/student4.txt;
	//Export导出到HDFS上
	hive (default)> export table default.student to '/user/hive/warehouse/export/student';
	
	清除表中数据（Truncate）
	注意：Truncate只能删除管理表，不能删除外部表中数据
	hive (default)> truncate table student;


### 查询
	
	基本查询（Select…From） 
	//全表和特定字段查询
	hive (default)> select * from emp;
	hive (default)> select empno, ename from emp;
	//列别名
	hive (default)> select ename AS name, deptno dn from emp;
	//算术运算符
	//查询出所有员工的薪水后加1显示。
	hive (default)> select sal +1 from emp;
	//常用函数
	hive (default)> select count(*) cnt from emp;总行数
	hive (default)> select max(sal) max_sal from emp;最大值
	hive (default)> select min(sal) min_sal from emp;最小值
	hive (default)> select sum(sal) sum_sal from emp;总和
	hive (default)> select avg(sal) avg_sal from emp;平均值
	hive (default)> select * from emp limit 5; limit语句
	
	Where语句
	//查询出薪水大于1000的所有员工
	hive (default)> select * from emp where sal >1000;
	//比较运算符（Between/In/ Is Null）
	hive (default)> select * from emp where sal =5000;
	hive (default)> select * from emp where sal between 500 and 1000;
	hive (default)> select * from emp where comm is null;
	hive (default)> select * from emp where sal IN (1500, 5000);
	// Like和RLike
	//RLIKE子句是Hive中这个功能的一个扩展，其可以通过Java的正则表达式这个更强大的语言来指定匹配条件。
	//查找以2开头薪水的员工信息
	hive (default)> select * from emp where sal LIKE '2%';
	//找第二个数值为2的薪水的员工信息
	hive (default)> select * from emp where sal LIKE '_2%';
	//查找薪水中含有2的员工信息
	hive (default)> select * from emp where sal RLIKE '[2]';
	逻辑运算符（And/Or/Not）
	hive (default)> select * from emp where sal>1000 and deptno=30;
	hive (default)> select * from emp where sal>1000 or deptno=30;
	hive (default)> select * from emp where deptno not IN(30, 20);
	
	分组
	Group By语句
	//计算emp表每个部门的平均工资
	hive (default)> select t.deptno, avg(t.sal) avg_sal from emp t group by t.deptno;
	//计算emp每个部门中每个岗位的最高薪水
	hive (default)> select t.deptno, t.job, max(t.sal) max_sal from emp t group by t.deptno, t.job;
	
	Having语句
	having与where不同点
	//where针对表中的列发挥作用，查询数据；having针对查询结果中的列发挥作用，筛选数据。
	//having只用于group by分组统计语句。
	
	求每个部门的平均工资
	hive (default)> select deptno, avg(sal) from emp group by deptno;
	
	//求每个部门的平均薪水大于2000的部门
	hive (default)> select deptno, avg(sal) avg_sal from emp group by deptno having avg_sal > 2000;
	
	Join语句
	//等值Join,Hive支持通常的SQL JOIN语句，但是只支持等值连接，不支持非等值连接。
	//根据员工表和部门表中的部门编号相等，查询员工编号、员工名称和部门编号；
	hive (default)> select e.empno, e.ename, d.deptno, d.dname from emp e join dept d on e.deptno = d.deptno;
	//合并员工表和部门表
	hive (default)> select e.empno, e.ename, d.deptno from emp e join dept d on e.deptno = d.deptno;
	//内连接：只有进行连接的两个表中都存在与连接条件相匹配的数据才会被保留下来。
	hive (default)> select e.empno, e.ename, d.deptno from emp e join dept d on e.deptno = d.deptno;
	//左外连接：JOIN操作符左边表中符合WHERE子句的所有记录将会被返回。
	hive (default)> select e.empno, e.ename, d.deptno from emp e left join dept d on e.deptno = d.deptno;
	//右外连接：JOIN操作符右边表中符合WHERE子句的所有记录将会被返回。
	hive (default)> select e.empno, e.ename, d.deptno from emp e right join dept d on e.deptno = d.deptno;
	//满外连接：将会返回所有表中符合WHERE语句条件的所有记录。如果任一表的指定字段没有符合条件的值的话，那么就使用NULL值替代。
	hive (default)> select e.empno, e.ename, d.deptno from emp e full join dept d on e.deptno = d.deptno;
	
	//大多数情况下，Hive会对每对JOIN连接对象启动一个MapReduce任务。本例中会首先启动一个MapReduce job对表e和表d进行连接操作，然后会再启动一个MapReduce job将第一个MapReduce job的输出和表l;进行连接操作。
	//注意：为什么不是表d和表l先进行连接操作呢？这是因为Hive总是按照从左到右的顺序执行的。
	hive (default)>SELECT e.ename, d.deptno, l. loc_name
	FROM   emp e 
	JOIN   dept d
	ON     d.deptno = e.deptno 
	JOIN   location l
	ON     d.loc = l.loc;
	
	//笛卡尔积 JOIN
	hive (default)> select empno, deptno from emp, dept;
	FAILED: SemanticException Column deptno Found in more than One Tables/Subqueries
	
	排序
	//全局排序（Order By）：全局排序，一个MapReduce
	//查询员工信息按工资升序排列
	hive (default)> select * from emp order by sal;
	//查询员工信息按工资降序排列
	hive (default)> select * from emp order by sal desc;
	//按照员工薪水的2倍排序
	hive (default)> select ename, sal*2 twosal from emp order by twosal;
	//按照部门和工资升序排序
	hive (default)> select ename, deptno, sal from emp order by deptno, sal ;
	
	//每个MapReduce内部排序（Sort By）:
	//Sort By：每个MapReduce内部进行排序，对全局结果集来说不是排序
	//根据部门降序查看员工信息
	hive (default)> select * from emp sort by empno desc;
	
	//分区排序（Distribute By）：类似MR中partition，进行分区，结合sort by使用。
	//注意，Hive要求DISTRIBUTE BY语句要写在SORT BY语句之前。
	//（1）先按照部门编号分区，再按照员工编号降序排序。
	hive (default)> insert overwrite local directory '/opt/module/datas/distby-desc' select * from emp distribute by deptno sort by empno desc;
	
	Cluster By
	//当distribute by和sorts by字段相同时，可以使用cluster by方式。
	//cluster by除了具有distribute by的功能外还兼具sort by的功能。但是排序只能是倒序排序，不能指定排序规则为ASC或者DESC。
	select * from emp cluster by deptno;等价于
	select * from emp distribute by deptno sort by deptno;
	
	分桶及抽样查询
	//分区针对的是数据的存储路径；分桶针对的是数据文件。



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

