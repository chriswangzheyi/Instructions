#HBase操作

##说明

HBase是一个分布式的、面向列的开源数据库。Hbase的组成结构可用表形容:

![](../Images/1.png)

##操作步骤

##删除Hadoop 损坏的文件(非必须)

 hadoop fsck -delete /

###前置操作

关闭hadoop安全模式

	hdfs dfsadmin -safemode leave

删掉不必要文件（针对 org.apache.zookeeper.KeeperException$NoAuthException: KeeperErrorCode KeeperErrorCode = NoNode for (节点路径)。）

	rm -rf /root/zookeeper-3.4.14/temp/zookeeper/version-2


### 建表 

    create 'table', 'column_family_1','column_family_2','column_family_3'...

创表的关键字是create，”hbase_test”是表名；”column_family_1”，”column_family_2”，”column_family_3”是三个不同的列族名。

例子：create 'User','address','info','member_id'


### 列出所有表

	list

### 查看描述信息

	describe '表名'

### 禁用表

	desable '表名'

### 启用表

	enable '表名'

### 是否启用

	 is_enabled '表名'

### 删除族列

**删除族列前需要禁用表**

	alter '表名', {NAME=>'列名', METHOD=>'delete'}
	
	#例子
	alter 'User', {NAME=>'member_id', METHOD=>'delete'}

**删除族列后需要启用表**


删除前：

![](../Images/2.png)

删除后：

![](../Images/3.png)


### 删除一张表

**删表前需要禁用表**

	drop '表名'

### 查询表是否存在

	exists '表名'

### 插入数据

	put 'User', 'zhangsan','address:country','China'
	put 'User', 'zhangsan','address:city','Chengdu'
	put 'User', 'zhangsan','info:age','20'
	put 'User', 'zhangsan','info:sex','female'
	put 'User', 'wangqiang','address:country','China'
	put 'User', 'wangqiang','address:city','chongqing'
	put 'User', 'wangqiang','info:age','22'
	put 'User', 'wangqiang','info:sex','male'	

### 查询数据

	# 获取所有第二族列叫'zhangsan'的数据
	get 'User','zhangsan'

	
	# 获取所有第二族列叫'zhangsan'并且数据前缀为info的数据
	get 'User','zhangsan','info'

	# 获取所有第二族列叫'zhangsan'并且数据前缀为info:age的数据
	get 'User','zhangsan','info:age'


### 更新数据

更新数据只能覆盖

	put 'User', 'zhangsan','info:age','19'

### 查看族列下某一项的所有信息

	scan 'User',{COLUMNS=>'info:age'}


### 通过timestamp来获取信息

	get 'User','zhangsan', {COLUMNS=>'info:age' ,TIMESTAMP=>'1571968111600'}

### 全表扫描

	scan 'User'


### 查询有多少行

	count 'User'

### 删除某人的全部信息

	deleteall 'User', 'wangqiang'