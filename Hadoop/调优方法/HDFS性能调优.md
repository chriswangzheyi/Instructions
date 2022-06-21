# HDFS性能调优

## 调整默认block大小

	<property>
	<name>dfs.blocksize</name>
	<value>30m</value>
	</property>

## 内存调优

	vim hadoop-env.sh
	export HADOOP_NAMENODE_OPTS="-Xmx512m -Xms256m -Dhadoop.security.logger=${HADOOP_SECURITY_LOGGER:-INFO,RFAS} -Dhdfs.audit.logger=${HDFS_AUDIT_LOGGER:-INFO,NullAppender} $HADOOP_NAMENODE_OPTS"
	export HADOOP_DATANODE_OPTS="-Xmx512m -Xms256m -Dhadoop.security.logger=ERROR,RFAS $HADOOP_DATANODE_OPTS"
	export HADOOP_SECONDARYNAMENODE_OPTS="-Xmx512m -Xms256m  -Dhadoop.security.logger=${HADOOP_SECURITY_LOGGER:-INFO,RFAS} -Dhdfs.audit.logger=${HDFS_AUDIT_LOGGER:-INFO,NullAppender} $HADOOP_SECONDARYNAMENODE_OPTS"
	export HADOOP_NFS3_OPTS="$HADOOP_NFS3_OPTS"
	export HADOOP_PORTMAP_OPTS="-Xmx512m $HADOOP_PORTMAP_OPTS"
	export HADOOP_CLIENT_OPTS="-Xmx512m $HADOOP_CLIENT_OPTS"

###修改hdfs启动内存

	cd /opt/hadoop/hadoop-2.6.5/libexec
	vim hadoop-config.sh
	JAVA_HEAP_MAX=-Xmx512m

###修改yarn启动内存

	cd /opt/hadoop/hadoop-2.6.5/etc/hadoop
	vim yarn-env.sh
	JAVA_HEAP_MAX=-Xmx512m
	YARN_HEAPSIZE=512

## 扩充hdfs磁盘

（1）dfs.data.dir

	<property>
	    <name>dfs.data.dir</name>
	    <value>/data/hadoop/hdfs,/test/hdfs</value>
	</property>


（2）在修改配置后重启hadoop集群即生效
