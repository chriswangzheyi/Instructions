# 04搭建Hive

## 解压

	tar -zxvf apache-hive-2.2.0-bin.tar.gz
	mv apache-hive-2.2.0-bin /usr/local/hive

## 拷贝mysql驱动

cp /root/mysql-connector-java-8.0.18.jar /usr/local/hive/lib

## 配置
	
	cd /usr/local/hive/conf
	cp hive-default.xml.template hive-site.xml
	echo "" > hive-site.xml

	vim hive-site.xml

插入：

	<?xml version="1.0"?>
	<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
	<configuration>
	    <property>
	      <name>javax.jdo.option.ConnectionURL</name>
		  <value>jdbc:mysql://192.168.195.150:3306/hive?createDatabaseIfNotExist=true</value>
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
	</configuration>


### 配置hive-env.sh

	cd /usr/local/hive/conf

	cp hive-env.sh.template hive-env.sh

修改

	vi hive-env.sh

添加

	export HADOOP_HOME=/usr/local/hadoop
	export HIVE_CONF_DIR=/usr/local/hive/conf

## 配置数据库

	mysql -uroot -p
	
	create database hive;



### 环境配置

	vi /etc/profile

插入

	export HIVE_HOME=/usr/local/hive
	export PATH=$PATH:$HIVE_HOME/bin 

刷新

	source /etc/profile


## 启动

	hive --service metastore & schematool -dbType mysql -initSchema
	hive

