# MergeTree

## 简介 

MergeTree 在写入一批数据时，数据总会以数据片段的形式写入磁 盘，且数据片段不可修改。为了避 免片段过多，ClickHouse 会通过后 台线程，定期合并这些数据片段，属于相同分区的数据片段会被合 成 一个新的片段。这种数据片段往复合并的特点，也正是合并树名称的由来。

## 基本语法

### 建表

	create table tb_merge_tree( id Int8 ,
	name String ,
	ctime Date
	) engine=MergeTree() order by id
	partition by name ;


### 插入数据

	insert into tb_merge_tree values (1,'hng','2020-08-07'),(4,'hng','2020-08-07'),(3,'ada','2020-08-07'),(2,'ada','2020-08-07') ;
	
	
##  ReplacingMergeTree

这个引擎是在 MergeTree 的基础上，添加了“处理重复数据”的功能，该引擎和 MergeTree 的不同之处在于它会删除具有相同主键的重复项。数据的去重只会在合并的过程中出现。合并会在未知的 时间在后台进行，所以你无法预先作出计划。有一些数据可能仍未被处理。因此，ReplacingMergeTree适用于在后台清除重复的数据以节省空间，但是它不保证没有重复的数据出现。


### 例子

创建表

	create table test_replacingMergeTree( id Int8 ,
	name String ,
	gender String ,
	city String
	)engine = ReplacingMergeTree()
	order by name
	partition by city ;
	
	# 插入收据
	insert into test_replacingMergeTree values
	(1,'zss','1','北京'), (1,'zss','3','北京'), (1,'zss','2','北京'), (1,'lss','1','南京'), (1,'lss','2','南京'), (1,'lss','2','北京');
	
	# 查询表中的数据
	┌─id─┬─name─┬─gender─┬─city─┐ 
	│ 1│lss │2 │北京│
	│ 1│zss │1 │北京│
	│ 1│zss │3 │北京│
	│ 1│zss │2 │北京│ 
	└────┴──────┴──────┘ 
	
	┌─id─┬─name─┬─gender─┬─city─┐ 
	│ 1│lss │1 │南京│
	│ 1│lss │2 │南京│ 
	└────┴──────┴──────┴┘
	
	
	#合并表中的数据
	optimize table test_replacingMergeTree final ;
	
	┌─id─┬─name─┬─gender─┬─city─┐
	│ 1│lss │2 │北京│
	│ 1│zss │2 │北京│
	└────┴──────┴──────┘
	┌─id─┬─name─┬─gender─┬─city─┐
	│ 1│lss │2 │南京│
	└────┴──────┴──────┘

数据会在分区中排序 , 相同的排序字段是去除重复数据的依据, 并不是主键 也就是说表中的数据不能保证一定按照这个维度去重 ,因为这个维度中的数据并不一定在同一个。


在知道了ReplacingMergeTree 的使用方法后，现在简单梳理一下它的处理逻辑。

（1）使用ORBER BY 排序键作为判断重复数据的唯一键。

（2）只有在合并分区的时候才会触发删除重复数据的逻辑。

（3）以数据分区为单位删除重复数据。当分区合并时，同一分区内的重复数据会被删除；不同分区之间的重复数据不会被删除。

（4）在进行数据去重时，因为分区内的数据已经基于ORBER BY 进行了排序，所以能够找到那些相邻的重复数据。

（5）数据去重策略有两种：

-如果没有设置ver 版本号，则保留同一组重复数据中的最后一行。

-如果设置了ver 版本号，则保留同一组重复数据中ver 字段取值最大的那一行。


## CollapsingMergeTree

假设现在需要设计一款数据库，该数据库支持对已经存在的数据实现行级粒度的修改或删除，你会怎么设计？一种最符合常理的思维可能是：首先找到保存数据的文件，接着修改这个文件，删除或者修改那些需要变化的数据行。

然而在大数据领域，对于ClickHouse 这类高性能分析型数据库而言，对
数据源文件修改是一件非常奢侈且代价高昂的操作。相较于直接修改源文件，它们会将修改和删除操作转换成新增操作，即以增代删。

CollapsingMergeTree 就是一种通过以增代删的思路，支持行级数据修改和删除的表引擎。它通过定义一个sign 标记位字段，记录数据行的状态。如果sign 标记为1，则表示这是一行有效的数据；如果sign 标记为-1，则表示这行数据需要被删除。当CollapsingMergeTree 分区合并时，同一数据分区内，sign 标记为1 和-1 的一组数据会被抵消删除。这种1 和-1 相互抵消的操作，犹如将一张瓦楞纸折叠了一般。

### 创建CollapsingMergeTree 表

	CREATE TABLE tb_cps_merge_tree
	(
	user_id UInt64,
	name String,
	age UInt8,
	sign Int8
	)
	ENGINE = CollapsingMergeTree(sign)
	ORDER BY user_id;


插入状态行，注意sign 一列的值为1

	INSERT INTO tb_cps_merge_tree VALUES (1001,'ADA', 18, 1);

插入一行取消行，用于抵消上述状态行。注意sign 一列的值为-1，其余值与状态行一致；并且插入一行主键相同的新状态行

	INSERT INTO tb_cps_merge_tree VALUES (1001, 'ADA', 18, -1), (1001, 'MADA', 19, 1);


查询数据

	┌─user_id─┬─name─┬─age─┬─sign─┐
	│    1001 │ ADA  │  18 │    1 │
	└─────────┴──────┴─────┴──────┘
	┌─user_id─┬─name─┬─age─┬─sign─┐
	│    1001 │ ADA  │  18 │   -1 │
	│    1001 │ MADA │  19 │    1 │
	└─────────┴──────┴─────┴──────┘

优化数据

	optimize table tb_cps_merge_tree ;

查询

	┌─user_id─┬─s_age─┐
	│    1001 │    19 │
	└─────────┴───────┘


**注意:**

CollapsingMergeTree 虽然解决了主键相同的数据即时删除的问题，但是状态持续变化且多线程并行写入情况下，状态行与取消行位置可能乱序，导致无法正常折叠。只有保证老的状态行在在取消行的上面, 新的状态行在取消行的下面! 但是多线程无法保证写的顺序!

例子：

	CREATE TABLE UAct_order
	(
	UserID UInt64,
	PageViews UInt8,
	Duration UInt8,
	Sign Int8
	)ENGINE = CollapsingMergeTree(Sign)
	ORDER BY UserID;


#### 先插入取消行

	INSERT INTO UAct_order VALUES (4324182021466249495, 5, 146, -1);

#### 后插入状态行

	INSERT INTO UAct_order VALUES (4324182021466249495, 5, 146, 1);

#### 优化数据

	optimize table UAct_order;

#### 查询

	select * from UAct_order;


	┌──────────────UserID─┬─PageViews─┬─Duration─┬─Sign─┐
	│ 4324182021466249495 │         5 │      146 │   -1 │
	│ 4324182021466249495 │         5 │      146 │    1 │
	└─────────────────────┴───────────┴──────────┴──────┘

## VersionedCollapsingMergeTree

取消字段和数据版本同事使用,避免取消行数据无法删除的问题


为了解决CollapsingMergeTree 乱序写入情况下无法正常折叠问题，VersionedCollapsingMergeTree 表引擎在建表语句中新增了一列Version，用于在乱序情况下记录状态行与取消行的对应关系。主键相同，且Version 相同、Sign 相反的行，在Compaction 时会被删除。


	CREATE TABLE tb_vscmt
	(
	uid UInt64,
	name String,
	age UInt8,
	sign Int8,
	version UInt8
	)
	ENGINE = VersionedCollapsingMergeTree(sign, version)
	ORDER BY uid;


先插入一行取消行，注意Signz=-1, Version=1

	INSERT INTO tb_vscmt VALUES (1001, 'ADA', 18, -1, 1);

后插入一行状态行

	INSERT INTO tb_vscmt VALUES (1001, 'ADA', 18, 1, 1),(101, 'DAD', 19, 1, 2);(101, 'DAD', 11, 1, 3); 

数据版本

	INSERT INTO tb_vscmt VALUES(101, 'DAD', 11, 1, 3) ;

查询合并前数据
	
	┌──uid─┬─name─┬─age─┬─sign─┬─version─┐
	│ 1001 │ ADA  │  18 │   -1 │       1 │
	└──────┴──────┴─────┴──────┴─────────┘
	┌──uid─┬─name─┬─age─┬─sign─┬─version─┐
	│  101 │ DAD  │  19 │    1 │       2 │
	│ 1001 │ ADA  │  18 │    1 │       1 │
	└──────┴──────┴─────┴──────┴─────────┘
	┌─uid─┬─name─┬─age─┬─sign─┬─version─┐
	│ 101 │ DAD  │  11 │    1 │       3 │
	└─────┴──────┴─────┴──────┴─────────┘


合并数据

	optimize table tb_vscmt;

得到：

	┌─uid─┬─name─┬─age─┬─sign─┬─version─┐
	│ 101 │ DAD  │  19 │    1 │       2 │
	│ 101 │ DAD  │  11 │    1 │       3 │
	└─────┴──────┴─────┴──────┴─────────┘


## SummingMergeTree

该引擎继承自 MergeTree。区别在于，当合并 SummingMergeTree 表的数据片段时，ClickHouse 会把所有具有相同主键的行合并为一行，该行包含了被合并的行中具有数值数据类型的列的汇总值。如果主键的组合方式使得单个键值对应于大量的行，则可以显著的减少存储空间并加快数据查询的速度。


### 例子


	CREATE TABLE summing_table(
		id String,
		city String,
		sal UInt32,
		comm Float64,
		ctime DateTime
	)ENGINE = SummingMergeTree()
	PARTITION BY toYYYYMM(ctime)
	ORDER BY (id, city)
	PRIMARY KEY id ;



	insert into summing_table
	values
	(1,'shanghai',10,20,'2020-12-12 01:11:12'),(1,'shanghai',20,30,'2020-12-12 01:11:12'),
	(3,'shanghai',10,20,'2020-11-12 01:11:12'),(3,'Beijing',10,20,'2020-11-12 01:11:12') ;

查看

	select * from summing_table;
	
	┌─id─┬─city─────┬─sal─┬─comm─┬───────────────ctime─┐
	│ 3  │ Beijing  │  10 │   20 │ 2020-11-12 01:11:12 │
	│ 3  │ shanghai │  10 │   20 │ 2020-11-12 01:11:12 │
	└────┴──────────┴─────┴──────┴─────────────────────┘
	┌─id─┬─city─────┬─sal─┬─comm─┬───────────────ctime─┐
	│ 1  │ shanghai │  10 │   20 │ 2020-12-12 01:11:12 │
	│ 1  │ shanghai │  20 │   30 │ 2020-12-12 01:11:12 │
	└────┴──────────┴─────┴──────┴─────────────────────┘

### 优化表结构 

	OPTIMIZE TABLE summing_table FINAL;

查看

	select * from summing_table;


	┌─id─┬─city─────┬─sal─┬─comm─┬───────────────ctime─┐
	│ 3  │ Beijing  │  10 │   20 │ 2020-11-12 01:11:12 │
	│ 3  │ shanghai │  10 │   20 │ 2020-11-12 01:11:12 │
	└────┴──────────┴─────┴──────┴─────────────────────┘
	┌─id─┬─city─────┬─sal─┬─comm─┬───────────────ctime─┐
	│ 1  │ shanghai │  30 │   50 │ 2020-12-12 01:11:12 │
	└────┴──────────┴─────┴──────┴─────────────────────┘


相同分区中,相同的Id 和city 的数据会被聚合在一起，数字字段的数据都会被sum 在一起。


### 如果指定合并字段

只有指定的字段才会sum操作

	drop table summing_table2 ;
	CREATE TABLE summing_table2(
		id String,
		city String,
		sal UInt32,
		comm Float64,
		ctime DateTime
	)ENGINE = SummingMergeTree(sal)
	PARTITION BY toYYYYMM(ctime)
	ORDER BY (id, city)
	PRIMARY KEY id ;

插入：

	insert into summing_table2
	values
	(1,'shanghai',10,20,'2020-12-12 01:11:12'),(1,'shanghai',20,30,'2020-12-12 01:11:12'),
	(3,'shanghai',10,20,'2020-11-12 01:11:12'),(3,'Beijing',10,20,'2020-11-12 01:11:12') ;

查看：

	select * from summing_table2;


	┌─id─┬─city─────┬─sal─┬─comm─┬───────────────ctime─┐
	│ 3  │ Beijing  │  10 │   20 │ 2020-11-12 01:11:12 │
	│ 3  │ shanghai │  10 │   20 │ 2020-11-12 01:11:12 │
	└────┴──────────┴─────┴──────┴─────────────────────┘
	┌─id─┬─city─────┬─sal─┬─comm─┬───────────────ctime─┐
	│ 1  │ shanghai │  10 │   20 │ 2020-12-12 01:11:12 │
	│ 1  │ shanghai │  20 │   30 │ 2020-12-12 01:11:12 │
	└────┴──────────┴─────┴──────┴─────────────────────┘


优化表

	OPTIMIZE TABLE summing_table2 FINAL;


	┌─id─┬─city─────┬─sal─┬─comm─┬───────────────ctime─┐
	│ 3  │ Beijing  │  10 │   20 │ 2020-11-12 01:11:12 │
	│ 3  │ shanghai │  10 │   20 │ 2020-11-12 01:11:12 │
	└────┴──────────┴─────┴──────┴─────────────────────┘
	┌─id─┬─city─────┬─sal─┬─comm─┬───────────────ctime─┐
	│ 1  │ shanghai │  30 │   20 │ 2020-12-12 01:11:12 │
	└────┴──────────┴─────┴──────┴─────────────────────┘


可以看到，因为指定了SummingMergeTree(sal)，所以只有sal的数据被合并了。


## AggregatingMergeTree



AggregatingMergeTree 就有些许数据立方体的意思，它能够在合并分区的时候，按照预先定义的条件聚
合数据。同时，根据预先定义的聚合函数计算数据并通过二进制的格式存入表内。将同一分组下的多行数据聚合成一行，既减少了数据行，又降低了后续聚合查询的开销。可以说，AggregatingMergeTree是SummingMergeTree 的升级版，它们的许多设计思路是一致的，例如同时定义ORDER BY 与PRIMARY KEY 的原因和目的。但是在使用方法上，两者存在明显差异，应该说AggregatingMergeTree
的定义方式是MergeTree 家族中最为特殊的一个。


### 例子

	#建立明细表

	CREATE TABLE detail_table
	(id UInt8,
	ctime Date,
	uid UInt64
	) ENGINE = MergeTree()
	PARTITION BY toYYYYMM(ctime)
	ORDER BY (id, ctime);

	#插入明细数据
	INSERT INTO detail_table VALUES(1, '2020-08-06', 1);
	INSERT INTO detail_table VALUES(1, '2020-08-06', 2);
	INSERT INTO detail_table VALUES(2, '2020-08-07', 1);
	INSERT INTO detail_table VALUES(2, '2020-08-07', 2);

	#建立预先聚合表，
	CREATE TABLE agg_table
	(id UInt8,
	ctime Date,
	uid AggregateFunction(uniq, UInt64)
	) ENGINE = AggregatingMergeTree()
	PARTITION BY toYYYYMM(ctime)
	ORDER BY (id, ctime);

上面的语句表示在uniq这个字段上做聚合。

	INSERT INTO agg_table
	select id, ctime, uniqState(uid)
	from detail_table
	group by id, ctime ;