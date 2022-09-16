# Hive中常用SerDe
参考：https://blog.csdn.net/leen0304/article/details/125610016


## 概念

SerDe 是Serializer 和 Deserializer 的简称。它是 Hive用来处理记录并且将它们映射到 Hive 表中的字段数据类型。

## 常用的 SerDe

### LazySimpleSerDe

	org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe，

用来处理文本文件格式：TEXTFILE

	CREATE TABLE test_serde_lz STORED AS TEXTFILE AS
	SELECT  name
	FROM employee;
	
### ColumnarSerDe

	org.apache.hadoop.hive.serde2.columnar.ColumnarSerDe

用来处理 RCFile

	CREATE TABLE test_serde_cs 
	ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.columnar.ColumnarSerDe' 
	STORED AS RCFile AS
	SELECT  name
	FROM employee

### OrcSerde
	org.apache.hadoop.hive.ql.io.orc.OrcSerde 

用来处理 ORCFile

	CREATE TABLE test_serde_parquet
	STORED AS ORCFILE AS
	SELECT name from employee;


### ParquetHiveSerDe

	org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe

用来在 Hive 中读写 Parquet 数据格式的内置 SerDe

	CREATE TABLE test_serde_parquet
	STORED AS PARQUET AS
	SELECT name from employee;
	
### JSONSerDe

	org.openx.data.jsonserde.JsonSerDe

这是一个第三方的 SerDe，用来利用 Hive 读取 JSON 数据记录。

	CREATE TABLE test_serde_js(
	name string,
	sex string,
	age string
	)
	ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
	STORED AS TEXTFILE;
	
### HBaseSerDe

	org.apache.hadoop.hive.hbase.HBaseSerDe

内置的 SerDe，可以让 Hive 跟 HBase 进行集成。

	CREATE TABLE test_serde_hb(
	id string,
	name string,
	sex string,
	age string
	)
	ROW FORMAT SERDE 'org.apache.hadoop.hive.hbase.HBaseSerDe'
	STORED BY 'org.apache.hadoop.hive.hbase.HBaseStorageHandler'
	WITH SERDEPROPERTIES ("hbase.columns.mapping"=":key,info:name,info:sex,info:age")
	TBLPROPERTIES("hbase.table.name" = "test_serde");
