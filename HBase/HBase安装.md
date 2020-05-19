# HBase安装

https://blog.csdn.net/csdnliu123/article/details/105643450
https://blog.csdn.net/weixin_41443460/article/details/105776452

## 版本

hbase:2.2.4

hadoop:2.10.0

zookeeper:3.6.1



## 安装步骤

### 解压

	sudo tar -zxf /root/hbase-2.2.4-bin.tar.gz -C /root/
	
### 配置文件

	vi /etc/profile

插入

	export PATH=$PATH:/root/hbase-2.2.4/bin

更新

	source /etc/profile

### 验证

	hbase version


## 伪分布式模式配置


### 配置hbase-env.sh文件

	vi /root/hbase-2.2.4/conf/hbase-env.sh

插入

	export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.252.b09-2.el7_8.x86_64
	export HBASE_CLASSPATH=/root/hbase-2.2.4/conf
	export HBASE_LOG_DIR=/root/hbase-2.2.4/logs
	export HBASE_MANAGES_ZK= false # false用第三方zk


### 配置hbase-site.xml文件

	vi /root/hbase-2.2.4/conf/hbase-site.xml

插入

	 <configuration>
	        <property>
	            <name>hbase.rootdir</name>
	            <value>hdfs://172.18.156.87:9000/hbase</value>
	        </property>
	
	
	        <property>
	            <name>hbase.cluster.distributed</name>
	            <value>true</value>
	        </property>
	
	        <property>
	            <name>hbase.zookeeper.quorum</name>
	            <value>172.18.156.87</value>
	        </property>
	
	        <property>
	            <name>hbase.master.info.port</name>
	            <value>60010</value>
	        </property>
	
	        <property>
	        <name>hbase.unsafe.stream.capability.enforce</name>
	        <value>false</value>
	        </property>
	
	  </configuration>




## 启动


### 先启动zookeeper

	cd /root/apache-zookeeper-3.6.1-bin/bin
	./zkServer.sh start

### 先启动hadoop

	cd /root/hadoop-2.10.0/sbin/ 
	start-all.sh

### 再启动hbase

	cd /root/hbase-2.2.4/bin
	start-hbase.sh

## 验证

### 进入hbase的shell界面

	cd /root/hbase-2.2.4/bin
	hbase shell

## 停止

	cd /root/hbase-2.2.4/bin
	stop-hbase.sh


## 验证

管理界面

	http://47.112.142.231:60010/master-status



## 特殊情况处理

	error: KeeperErrorCode = NoNode for /hbase/master

考虑zk问题。


如果HBase初始化失败，进入zk客户端后，删除zk下的habase节点

	deleteall /hbase