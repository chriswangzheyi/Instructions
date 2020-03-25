# MySQL索引优化

## 索引类型


可以使用SHOW INDEX FROM table_name;查看索引详情：

### 主键索引 PRIMARY KEY

它是一种特殊的唯一索引，不允许有空值。一般是在建表的时候同时创建主键索引。注意：一个表只能有一个主键。

### 唯一索引 UNIQUE

唯一索引列的值必须唯一，但允许有空值。如果是组合索引，则列值的组合必须唯一。

可以通过ALTER TABLE table_name ADD UNIQUE (column);创建唯一索引：

可以通过ALTER TABLE table_name ADD UNIQUE (column1,column2);创建唯一组合索引：

### 普通索引 INDEX

这是最基本的索引，它没有任何限制。

可以通过ALTER TABLE table_name ADD INDEX index_name (column);创建普通索引：

###组合索引 INDEX

即一个索引包含多个列，多用于避免回表查询。

可以通过ALTER TABLE table_name ADD INDEX index_name(column1,column2, column3);创建组合索引：

### 全文索引 FULLTEXT

也称全文检索，是目前搜索引擎使用的一种关键技术。

可以通过ALTER TABLE table_name ADD FULLTEXT (column);创建全文索引：

索引一经创建不能修改，如果要修改索引，只能删除重建。可以使用DROP INDEX index_name ON table_name;删除索引。


## 索引设计的原则 

适合索引的列是出现在where子句中的列，或者连接子句中指定的列；

基数较小的类，索引效果较差，没有必要在此列建立索引；

使用短索引，如果对长字符串列进行索引，应该指定一个前缀长度，这样能够节省大量索引空间；

不要过度索引。索引需要额外的磁盘空间，并降低写操作的性能。在修改表内容的时候，索引会进行更新甚至重构，索引列越多，这个时间就会越长。所以只保持需要的索引有利于查询即可。


## 索引失效

1.**如果条件中有or**，即使其中有条件带索引也不会使用(这也是为什么尽量少用or的原因)

2.对于多列索引，不是使用的第一部分(第一个)，则不会使用索引 **（靠左原则）**

3.l**ike查询是以%开头**

4.如果列类型是字符串，那一定要在条件中将数据使用引号引用起来,否则不使用索引

5.如果mysql估计使用全表扫描要比使用索引快,则不使用索引

6 不等于(！= ，<> )，EXISTS，not in,is  not null,>,<都会失效，in（in里面包含了子查询）（非主键索引）