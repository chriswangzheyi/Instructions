# Hive 安装

https://blog.csdn.net/Aeve_imp/article/details/105874934

## 版本

Hadoop 2.10.0

Hive 2.3.7

Mysql 8.0.20 

## 安装步骤

### 前提条件

安装好hadoop和mysql

### 解压

	tar -zxvf apache-hive-2.3.7-bin.tar.gz -C /root

### 环境配置

	vi /etc/profile

插入

	export HIVE_HOME=/root/apache-hive-2.3.7-bin
	export PATH=$PATH:$HIVE_HOME/bin 

刷新

	source /etc/profile


### 配置hive-env.sh

	cd /root/apache-hive-2.3.7-bin/conf

	cp hive-env.sh.template hive-env.sh

修改

	vi hive-env.sh

添加

	export HADOOP_HOME=/root/hadoop-2.10.0/
	export HIVE_CONF_DIR=/root/apache-hive-2.3.7-bin/conf


### 配置hive-site.xml

	cd /root/apache-hive-2.3.7-bin/conf
	vi hive-site.xml

插入：

	<?xml version="1.0"?>
	<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
	<configuration>
		<property>
		  <name>javax.jdo.option.ConnectionURL</name>
		  <value>jdbc:mysql://172.18.156.87:3306/hive?useSSL=false</value>
		  <description>JDBC connect string for a JDBC metastore</description>
		</property>
	
		<property>
		  <name>javax.jdo.option.ConnectionDriverName</name>
		  <value>com.mysql.cj.jdbc.Driver</value>
		  <description>Driver class name for a JDBC metastore</description>
		</property>
	
		<property>
		  <name>javax.jdo.option.ConnectionUserName</name>
		  <value>root</value>
		  <description>username to use against metastore database</description>
		</property>
	
		<property>
		  <name>javax.jdo.option.ConnectionPassword</name>
		  <value>1qa2ws#ED</value>
		  <description>password to use against metastore database</description>
		</property>
		<property>
			<name>hive.metastore.warehouse.dir</name>
			<value>/user/hive/warehouse</value>
			<description>location of default database for the warehouse</description>
		</property>
		
		<property>
		    <name>hive.metastore.schema.verification</name>
		    <value>false</value>
		</property>
		
		<property>
		    <name>datanucleus.schema.autoCreateAll</name>
		    <value>true</value>
		 </property>		
	</configuration>


注意：

ConnectionPassword 就是Mysql的连接密码


### 拷贝mysql驱动

cp /root/mysql-connector-java-8.0.18.jar /root/apache-hive-2.3.7-bin/lib


### 在hadoop中建文件夹

	hdfs dfs -mkdir /tmp
	hdfs dfs -mkdir -p /user/hive/warehouse
	hdfs dfs -chmod g+w /tmp
	hdfs dfs -chmod g+w /user/hive/warehouse


## 创建数据库

	mysql -uroot -p
	
	create database hive;
	alter database hive character set latin1;
	
## 初始化元数据

	schematool -dbType mysql -initSchema	
	
显示下面内容表示成功：

	Initialization script completed
	schemaTool completed	


## 启动

	cd /root/apache-hive-2.3.7-bin/bin

用来启动metastore

	nohup hive --service metastore 2>&1 &

用来启动hiveserver2

	nohup  hive --service hiveserver2   2>&1 &



## 验证

	show databases;