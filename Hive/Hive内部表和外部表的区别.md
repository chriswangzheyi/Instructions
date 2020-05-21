# Hive内部表和外部表的区别

参考：https://blog.csdn.net/qq_36743482/article/details/78393678



## 内部表&外部表

未被external修饰的是内部表（managed table），被external修饰的为外部表（external table）；

### 区别：
内部表数据由Hive自身管理，外部表数据由HDFS管理；

内部表数据存储的位置是hive.metastore.warehouse.dir（默认：/user/hive/warehouse），外
部表数据的存储位置由自己制定（如果没有LOCATION，Hive将在HDFS上的/user/hive/warehouse文
件夹下以外部表的表名创建一个文件夹，并将属于这个表的数据存放在这里）；

删除内部表会直接删除元数据（metadata）及存储数据；删除外部表仅仅会删除元数据，HDFS上的文件并不会被删除；

对内部表的修改会将修改直接同步给元数据，而对外部表的表结构和分区进行修改，则需要修复（MSCK REPAIR TABLE table_name;）

## 试验理解


### 创建内部表t1

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


### 装载数据（t1）

	load data local inpath '/root/data/1.data'overwrite into table t1;


### 创建外部表t2

	create external table t2(
	    id      int
	   ,name    string
	   ,hobby   array<string>
	   ,add     map<String,string>
	)
	row format delimited
	fields terminated by ','
	collection items terminated by '-'
	map keys terminated by ':'
	location '/user/t2'
	;


### ### 装载数据（t2）

	load data local inpath '/root/data/1.data' overwrite into table t2;


##查看文件位置

### 外部表

![](../Images/3.png)

可以看到储存位置在 /usr目录下

### 内部表

![](../Images/4.png)

可以看到储存位置在 /user/hive/warehouse 目录下


## 命令行查看

### 查看表信息

	desc formatted t1;

![](../Images/5.png)

	desc formatted t2;

![](../Images/6.png)

**managed table就是内部表，而external table就是外部表**

### 区别

分别删除内部表和外部表后：

观察HDFS上的文件：

t1已经不存在了

t2还存在