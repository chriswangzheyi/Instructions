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


###建表 

    create 'table', 'column_family_1','column_family_2','column_family_3'...

创表的关键字是create，”hbase_test”是表名；”column_family_1”，”column_family_2”，”column_family_3”是三个不同的列族名。

例子：create 'User','address','info','member_id'