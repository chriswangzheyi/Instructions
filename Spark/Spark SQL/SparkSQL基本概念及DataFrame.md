# SparkSQL基本概念及DataFrame

## 简介

类似Hive之于Hadoop，SparkSQL将任务转化为RDD，提交到集群执行，效率非常快。用于处理结构化数据。



## 创建DataFrame

DataFrame可以理解成是一个表

### 通过Case class定义表

	#定义一个case class来代表emp表的schema结构
	scala> case class Emp(empno:Int,ename:String,job:String,mgr:String,hiredate:String,sal:Int,comm:String,deptno:Int)
	defined class Emp
	
	#导入emp.csv文件
	scala> val lines = sc.textFile("/home/hadoop_spark/emp.csv").map(_.split(","))
	lines: org.apache.spark.rdd.RDD[Array[String]] = MapPartitionsRDD[2] at map at <console>:24	

	#将case class和RDD关联起来
	scala> val allEmp = lines.map(x=>Emp(x(0).toInt,x(1),x(2),x(3),x(4),x(5).toInt,x(6),x(7).toInt))
	allEmp: org.apache.spark.rdd.RDD[Emp] = MapPartitionsRDD[3] at map at <console>:27

	#生成表 toDF是转换为DataFrame的意思
	scala> val empDF = allEmp.toDF
	empDF: org.apache.spark.sql.DataFrame = [empno: int, ename: string ... 6 more fields]
	
	#查看表
	scala> empDF.show
	+-----+------+---------+----+----------+----+----+------+                       
	|empno| ename|      job| mgr|  hiredate| sal|comm|deptno|
	+-----+------+---------+----+----------+----+----+------+
	| 7369| SMITH|    CLERK|7902|1980/12/17| 800|    |    20|
	| 7499| ALLEN| SALESMAN|7698| 1981/2/20|1600| 300|    30|
	| 7521|  WARD| SALESMAN|7698| 1981/2/22|1250| 500|    30|
	| 7566| JONES|  MANAGER|7839|  1981/4/2|2975|    |    20|
	| 7654|MARTIN| SALESMAN|7698| 1981/9/28|1250|1400|    30|
	| 7698| BLAKE|  MANAGER|7839|  1981/5/1|2850|    |    30|
	| 7782| CLARK|  MANAGER|7839|  1981/6/9|2450|    |    10|
	| 7788| SCOTT|  ANALYST|7566| 1987/4/19|3000|    |    20|
	| 7839|  KING|PRESIDENT|    |1981/11/17|5000|    |    10|
	| 7844|TURNER| SALESMAN|7698|  1981/9/8|1500|   0|    30|
	| 7876| ADAMS|    CLERK|7788| 1987/5/23|1100|    |    20|
	| 7900| JAMES|    CLERK|7698| 1981/12/3| 950|    |    30|
	| 7902|  FORD|  ANALYST|7566| 1981/12/3|3000|    |    20|
	| 7934|MILLER|    CLERK|7782| 1982/1/23|1300|    |    10|
	+-----+------+---------+----+----------+----+----+------+
	
	#查看表结构
	scala> empDF.printSchema
	root
	 |-- empno: integer (nullable = false)
	 |-- ename: string (nullable = true)
	 |-- job: string (nullable = true)
	 |-- mgr: string (nullable = true)
	 |-- hiredate: string (nullable = true)
	 |-- sal: integer (nullable = false)
	 |-- comm: string (nullable = true)
	 |-- deptno: integer (nullable = false)

### 通过SparkSession创建表

	#导入数据
	scala> val empCSV = sc.textFile("/home/hadoop_spark/emp.csv").map(_.split(","))
	empCSV: org.apache.spark.rdd.RDD[Array[String]] = MapPartitionsRDD[9] at map at <console>:24
	
	# 导入包
	scala> import org.apache.spark.sql._
	import org.apache.spark.sql._
	
	scala> import org.apache.spark.sql.types._
	import org.apache.spark.sql.types._
	
	#创建Type
	scala> val myschema = StructType(List(StructField("empno", DataTypes.IntegerType), StructField("ename", DataTypes.StringType),StructField("job", DataTypes.StringType),StructField("mgr", DataTypes.StringType),StructField("hiredate", DataTypes.StringType),StructField("sal", DataTypes.IntegerType),StructField("comm", DataTypes.StringType),StructField("deptno", DataTypes.IntegerType)))
	myschema: org.apache.spark.sql.types.StructType = StructType(StructField(empno,IntegerType,true), StructField(ename,StringType,true), StructField(job,StringType,true), StructField(mgr,StringType,true), StructField(hiredate,StringType,true), StructField(sal,IntegerType,true), StructField(comm,StringType,true), StructField(deptno,IntegerType,true))
	
	#把读入的数据empCSV映射一行Row：注意这里没有带结构
	scala> val rowRDD = empCSV.map(x=>Row(x(0).toInt,x(1),x(2),x(3),x(4),x(5).toInt,x(6),x(7).toInt))
	rowRDD: org.apache.spark.rdd.RDD[org.apache.spark.sql.Row] = MapPartitionsRDD[13] at map at <console>:31
	
	#通过SparkSession.createDataFrame(数据，schema结构)创建表
	scala> val df = spark.createDataFrame(rowRDD,myschema)
	df: org.apache.spark.sql.DataFrame = [empno: int, ename: string ... 6 more fields]
	
	#查看表结构
	scala> df.show
	+-----+------+---------+----+----------+----+----+------+
	|empno| ename|      job| mgr|  hiredate| sal|comm|deptno|
	+-----+------+---------+----+----------+----+----+------+
	| 7369| SMITH|    CLERK|7902|1980/12/17| 800|    |    20|
	| 7499| ALLEN| SALESMAN|7698| 1981/2/20|1600| 300|    30|
	| 7521|  WARD| SALESMAN|7698| 1981/2/22|1250| 500|    30|
	| 7566| JONES|  MANAGER|7839|  1981/4/2|2975|    |    20|
	| 7654|MARTIN| SALESMAN|7698| 1981/9/28|1250|1400|    30|
	| 7698| BLAKE|  MANAGER|7839|  1981/5/1|2850|    |    30|
	| 7782| CLARK|  MANAGER|7839|  1981/6/9|2450|    |    10|
	| 7788| SCOTT|  ANALYST|7566| 1987/4/19|3000|    |    20|
	| 7839|  KING|PRESIDENT|    |1981/11/17|5000|    |    10|
	| 7844|TURNER| SALESMAN|7698|  1981/9/8|1500|   0|    30|
	| 7876| ADAMS|    CLERK|7788| 1987/5/23|1100|    |    20|
	| 7900| JAMES|    CLERK|7698| 1981/12/3| 950|    |    30|
	| 7902|  FORD|  ANALYST|7566| 1981/12/3|3000|    |    20|
	| 7934|MILLER|    CLERK|7782| 1982/1/23|1300|    |    10|
	+-----+------+---------+----+----------+----+----+------+

### 直接读取一个具有格式的数据文件（Json文件）
前提：数据文件本身具有格式

	scala> val peopleDF = spark.read.json("/home/hadoop_spark/emp.json")
	peopleDF: org.apache.spark.sql.DataFrame = [comm: string, deptno: bigint ... 6 more fields]
	
	scala> peopleDF.show
	+----+------+-----+------+----------+---------+----+----+
	|comm|deptno|empno| ename|  hiredate|      job| mgr| sal|
	+----+------+-----+------+----------+---------+----+----+
	|    |    20| 7369| SMITH|1980/12/17|    CLERK|7902| 800|
	| 300|    30| 7499| ALLEN| 1981/2/20| SALESMAN|7698|1600|
	| 500|    30| 7521|  WARD| 1981/2/22| SALESMAN|7698|1250|
	|    |    20| 7566| JONES|  1981/4/2|  MANAGER|7839|2975|
	|1400|    30| 7654|MARTIN| 1981/9/28| SALESMAN|7698|1250|
	|    |    30| 7698| BLAKE|  1981/5/1|  MANAGER|7839|2850|
	|    |    10| 7782| CLARK|  1981/6/9|  MANAGER|7839|2450|
	|    |    20| 7788| SCOTT| 1987/4/19|  ANALYST|7566|3000|
	|    |    10| 7839|  KING|1981/11/17|PRESIDENT|    |5000|
	|   0|    30| 7844|TURNER|  1981/9/8| SALESMAN|7698|1500|
	|    |    20| 7876| ADAMS| 1987/5/23|    CLERK|7788|1100|
	|    |    30| 7900| JAMES| 1981/12/3|    CLERK|7698| 950|
	|    |    20| 7902|  FORD| 1981/12/3|  ANALYST|7566|3000|
	|    |    10| 7934|MILLER| 1982/1/23|    CLERK|7782|1300|
	+----+------+-----+------+----------+---------+----+----+


## 操作算子

### DSL（不常用）

查询所有的员工信息:  

	df.show

查询员工信息：姓名
			    
	scala> df.select("ename").show
	+------+
	| ename|
	+------+
	| SMITH|
	| ALLEN|
	|  WARD|
	| JONES|
	|MARTIN|
	| BLAKE|
	| CLARK|
	| SCOTT|
	|  KING|
	|TURNER|
	| ADAMS|
	| JAMES|
	|  FORD|
	|MILLER|
	+------+

				
查询员工信息： 姓名  薪水  薪水+100
			    
	scala> df.select($"ename",$"sal",$"sal"+100).show
	+------+----+-----------+
	| ename| sal|(sal + 100)|
	+------+----+-----------+
	| SMITH| 800|        900|
	| ALLEN|1600|       1700|
	|  WARD|1250|       1350|
	| JONES|2975|       3075|
	|MARTIN|1250|       1350|
	| BLAKE|2850|       2950|
	| CLARK|2450|       2550|
	| SCOTT|3000|       3100|
	|  KING|5000|       5100|
	|TURNER|1500|       1600|
	| ADAMS|1100|       1200|
	| JAMES| 950|       1050|
	|  FORD|3000|       3100|
	|MILLER|1300|       1400|
	+------+----+-----------+
				
查询工资大于2000的员工

	scala> df.filter($"sal" > 2000).show
	+-----+-----+---------+----+----------+----+----+------+
	|empno|ename|      job| mgr|  hiredate| sal|comm|deptno|
	+-----+-----+---------+----+----------+----+----+------+
	| 7566|JONES|  MANAGER|7839|  1981/4/2|2975|    |    20|
	| 7698|BLAKE|  MANAGER|7839|  1981/5/1|2850|    |    30|
	| 7782|CLARK|  MANAGER|7839|  1981/6/9|2450|    |    10|
	| 7788|SCOTT|  ANALYST|7566| 1987/4/19|3000|    |    20|
	| 7839| KING|PRESIDENT|    |1981/11/17|5000|    |    10|
	| 7902| FORD|  ANALYST|7566| 1981/12/3|3000|    |    20|
	+-----+-----+---------+----+----------+----+----+------+

分组：

	scala> df.groupBy($"deptno").count.show
	+------+-----+                                                                  
	|deptno|count|
	+------+-----+
	|    20|    5|
	|    10|    3|
	|    30|    6|
	+------+-----+
		

### SQL

注意：需要将DataFrame注册成一个视图(view)

	scala> df.createOrReplaceTempView("emp")
	
demo1:

	scala> spark.sql("select * from emp").show
	+-----+------+---------+----+----------+----+----+------+
	|empno| ename|      job| mgr|  hiredate| sal|comm|deptno|
	+-----+------+---------+----+----------+----+----+------+
	| 7369| SMITH|    CLERK|7902|1980/12/17| 800|    |    20|
	| 7499| ALLEN| SALESMAN|7698| 1981/2/20|1600| 300|    30|
	| 7521|  WARD| SALESMAN|7698| 1981/2/22|1250| 500|    30|
	| 7566| JONES|  MANAGER|7839|  1981/4/2|2975|    |    20|
	| 7654|MARTIN| SALESMAN|7698| 1981/9/28|1250|1400|    30|
	| 7698| BLAKE|  MANAGER|7839|  1981/5/1|2850|    |    30|
	| 7782| CLARK|  MANAGER|7839|  1981/6/9|2450|    |    10|
	| 7788| SCOTT|  ANALYST|7566| 1987/4/19|3000|    |    20|
	| 7839|  KING|PRESIDENT|    |1981/11/17|5000|    |    10|
	| 7844|TURNER| SALESMAN|7698|  1981/9/8|1500|   0|    30|
	| 7876| ADAMS|    CLERK|7788| 1987/5/23|1100|    |    20|
	| 7900| JAMES|    CLERK|7698| 1981/12/3| 950|    |    30|
	| 7902|  FORD|  ANALYST|7566| 1981/12/3|3000|    |    20|
	| 7934|MILLER|    CLERK|7782| 1982/1/23|1300|    |    10|
	+-----+------+---------+----+----------+----+----+------+

demo2:

	scala> spark.sql("select * from emp where deptno=10").show
	+-----+------+---------+----+----------+----+----+------+
	|empno| ename|      job| mgr|  hiredate| sal|comm|deptno|
	+-----+------+---------+----+----------+----+----+------+
	| 7782| CLARK|  MANAGER|7839|  1981/6/9|2450|    |    10|
	| 7839|  KING|PRESIDENT|    |1981/11/17|5000|    |    10|
	| 7934|MILLER|    CLERK|7782| 1982/1/23|1300|    |    10|
	+-----+------+---------+----+----------+----+----+------+
	
	
demo3:

	scala> spark.sql("select deptno,sum(sal) from emp group by deptno").show
	+------+--------+
	|deptno|sum(sal)|
	+------+--------+
	|    20|   10875|
	|    10|    8750|
	|    30|    9400|
	+------+--------+
