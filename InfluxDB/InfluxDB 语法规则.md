# InfluxDB 语法规则

## 查询版本号

	influx -precision rfc3339

## 数据库

### 创建数据库

	create database <table name>

例：

	create database test

### 删除数据库
	
	drop database <table name>

例：

	drop database test

## 表

### 显示所有表

	SHOW MEASUREMENTS

### 插入（不带时间戳）

	insert <measurement>[,<tag-key>=<tag-value>...] <field-key>=<field-value>[,<field2-key>=<field2-value>...] [unix-nano-timestamp] 


例子：

	insert test,host=127.0.0.1,monitor_name=test1,app=ios count=2,num=3

### 插入（自带时间戳）

	insert disk_free,hostname=server01 value=442221834240i 1435362189575692182	


### 保存策略

	create retention policy "rp_name" on "db_name" duration 3w replication 1 default

解释：

	rp_name：策略名；
	db_name：具体的数据库名；
	3w：保存3周，3周之前的数据将被删除，influxdb具有各种事件参数，比如：h（小时），d（天），w（星期）；
	replication 1：副本个数，一般为1就可以了；