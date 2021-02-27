# 03搭建Hbase

## 安装

	tar -zxvf hbase-2.2.6-bin.tar.gz
	mv hbase-2.2.6 /usr/local/hbase



## 配置

	cd /usr/local/hbase/conf
	vim hbase-site.xml

配置

	<configuration>
			<property>
			  <name>hbase.rootdir</name>
			  <value>hdfs://master:9000/hbase</value>
			</property>
			<property>
			  <name>hbase.cluster.distributed</name>
			  <value>true</value>
			</property>
			<property>
			  <name>hbase.zookeeper.quorum</name>
			  <value>master</value>
			</property>
			<property>
			  <name>dfs.replication</name>
			  <value>1</value>
			</property>
	</configuration>


vi hbase-env.sh

	export JAVA_HOME=/usr/local/jdk
	export HBASE_CLASSPATH=/usr/local/hbase/conf
	export HBASE_LOG_DIR=/usr/local/hbase/logs
	export HBASE_MANAGES_ZK= false # false用第三方zk


vim regionservers 

配置

	master


vim /etc/profile

配置

	export HBASE_HOME=/usr/local/hbase
	export PATH=$PATH:$JAVA_HOME/bin:$ZK_HOME/bin:$KAFKA_HOME/bin:$HADOOP_HOME/bin:$HBASE_HOME/bin


source /etc/profile

## 启动

	start-hbase.sh


## 验证

	http://192.168.195.150:16030/rs-status	

