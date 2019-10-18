#Hive 应用场景


参考资料： https://blog.csdn.net/weixin_43909426/article/details/86502507

##案例一

需求：现有这么一批数据，现要求出：每个用户截止到每月为止的最大单月访问次数和累计到该月的总访问次数。

	用户名，月份，访问次数
	A,2015-01,5
	A,2015-01,15
	B,2015-01,5
	A,2015-01,8
	B,2015-01,25
	A,2015-01,5
	A,2015-02,4
	A,2015-02,6
	B,2015-02,10
	B,2015-02,5
	A,2015-03,16
	A,2015-03,22
	B,2015-03,23
	B,2015-03,10
	B,2015-03,11

---
建表语句：

	create table t_user(
		name String,
		month String, 
		visitCount int
		) 
		ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';
		load data local inpath '/root/example1.txt'into table t_user;


执行语句：

	#step01 统计每个用户每月的总访问次数
	create view view_step01 as select name,month,sum(visitCount) total from t_user  group by name,month;
	#step02 （自连接，连接条件为name）
	create view view_step02 as select t1.name aname,t1.month amonth,t1.total atotal,t2.name bname,t2.month bmonth,t2.total btotal from view_step01 t1 join view_step01  t2 on t1.name =t2.name ;
	#step03 去除无用数据，每组找到小于等于自己月份的数据
	select bname,bmonth,max(btotal),sum(btotal),btotal from view_step02 where unix_timestamp(amonth,'yyyy-MM' )>=unix_timestamp(bmonth,'yyyy-MM') group by aname,amonth,atotal, bname, bmonth, btotal;



最终结果：


	用户  月份      最大访问次数  总访问次数       当月访问次数
	A     2015-01          33              33               33
	A     2015-02          33              43               10
	A     2015-03          38              81               38
	B     2015-01          30              30               30
	B     2015-02          30              45               15



![](../Images/1.png)


##案例二


需求：所有数学课程成绩 大于 语文课程成绩的学生的学号

准备数据

vi course.txt

	1,1,chinese,43
	2,1,math,55
	3,2,chinese,77
	4,2,math,88
	5,3,chinese,98
	6,3,math,65

建表语句：

	create table course (
	  id int,
	  sid int,
	  course String,
	  score int 
	)
	ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';
	load data local inpath '/root/course.txt'into table course;


id为主键， sid为学生id， 

解决：（行列转换）


	SELECT
	    t1.sid 
	FROM
	    (
	SELECT
	    sid,
	    max( CASE course WHEN "chinese" THEN score ELSE 0 END ) AS chinese,
	    max( CASE course WHEN "math" THEN score ELSE 0 END ) AS math 
	FROM
	    course 
	GROUP BY
	    sid 
	    ) t1 
	WHERE
	    t1.chinese < t1.math;


##案例三


需求：比如：2010012325表示在2010年01月23日的气温为25度。现在要求使用hive，计算每一年出现过的最大气温的日期+温度。
数据：

	年 温度
	20140101 14
	20140102 16
	20140103 17
	20140104 10
	20140105 06
	20120106 09
	20120107 32
	20120108 12
	20120109 19
	20120110 23
	20010101 16
	20010102 12
	20010103 10
	20010104 11
	20010105 29
	20130106 19
	20130107 22
	20130108 12
	20130109 29
	20130110 23
	20080101 05



准备数据

vi temperature.txt

	20140101 14
	20140102 16
	20140103 17
	20140104 10
	20140105 06
	20120106 09
	20120107 32
	20120108 12
	20120109 19
	20120110 23
	20010101 16
	20010102 12
	20010103 10
	20010104 11
	20010105 29
	20130106 19
	20130107 22
	20130108 12
	20130109 29
	20130110 23
	20080101 05


建表语句

	create table tmp (
		data String 
	)
	ROW FORMAT DELIMITED FIELDS TERMINATED BY '\n';
	load data local inpath '/root/temperature.txt'into table tmp;	


查询语句

	SELECT 
	substr(data, 1, 4 ) AS YEAR, 
	max( substr( data, 9, 11 ) ) AS temperature 
	FROM 
		tmp GROUP BY substr(data, 1, 4 );
	 