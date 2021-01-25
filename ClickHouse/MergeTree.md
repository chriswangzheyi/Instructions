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

数据会在分区中排序 , 相同的排序字段是去除重复数据的依据, 并不是主键 也就是说表中的数据不能保证一定按照这个维度去重 ,因为这个维度中的数据并不一定在同一个