#HBase安装


上传hbase-2.2.1-bin.tar.gz到服务器


解压

    tar -xzvf hbase-2.2.1-bin.tar.gz


修改hbase-env.sh配置文件

	vi /root/hbase-2.2.1/conf/hbase-env.sh


	#插入

	export HBASE_HOME=/root/hbase-2.2.1
	export JAVA_HOME=/root/jdk1.8.0_221
	export HADOOP_HOME=/root/hadoop-3.2.1
	export HBASE_LOG_DIR=$HBASE_HOME/logs
	export HBASE_PID_DIR=$HBASE_HOME/pids
	export HBASE_MANAGES_ZK=false


修改hbase-site.xml文件

	vi /root/hbase-2.2.1/conf/hbase-site.xml

	#插入

	<!--设置HRegionServers共享目录，mycluster是在Hadoop中设置的名字空间-->
	<property>
	        <name>hbase.rootdir</name>
	        <value>hdfs://mycluster/hbase</value>
	</property>
	<!--设置HMaster的rpc端口-->
	<property>
	        <name>hbase.master.port</name>
	        <value>16000</value>
	</property>
	<!--设置HMaster的http端口-->
	<property>
	        <name>hbase.master.info.port</name>
	        <value>16010</value>
	</property>
	<!--设置缓存文件存储路径-->
	<property>
	        <name>hbase.tmp.dir</name>
	        <value>/root/hbase-2.2.1/tmp</value>
	</property>
	<!--开启分布式模式-->
	<property>
	        <name>hbase.cluster.distributed</name>
	        <value>true</value>
	</property>
	<property>
	        <name>hbase.zookeeper.quorum</name>
	<!--指定zk的地址，多个用,分割-->
	        <value>Slave001,Slave002,Slave003</value>
	</property>	
	<!--指定zk端口-->
	<property>
	        <name>hbase.zookeeper.property.clientPort</name>
	        <value>2181</value>
	</property>	
	<!--指定zk数据目录，需要与zk集群的dataIdr配置一致-->
	<property>
	        <name>hbase.zookeeper.property.dataDir</name>
	        <value>/root/zookeeper-3.1.14/temp/zookeeper</value>
	</property>	


修改regionservers文件

	vi /root/hbase-2.2.1/conf/regionservers

	#插入

	Slave001
	Slave002
	Slave003

	#删除
	localhost
	

新建backup-masters文件

	vi /root/hbase-2.2.1/conf/backup-masters

	#插入

	Master002


在HBase安装目录下创建文件夹

	cd /root/hbase-2.2.1
	mkdir tmp logs pid


将HBase配置好的安装文件同步到其他节点

	#复制到Master002
	scp -r  /root/hbase-2.2.1  root@Master002:/root

	#复制到Slave001
	scp -r  /root/hbase-2.2.1  root@Slave001:/root

	#复制到Slave002
	scp -r  /root/hbase-2.2.1  root@Slave002:/root

	#复制到Slave003
	scp -r  /root/hbase-2.2.1  root@Slave003:/root
    

在各个节点上配置环境变量

	vi /etc/profile

	export HBASE_HOME=/root/hbase-2.2.1
	export PATH=$PATH:$HBASE_HOME/bin

	#使得配置生效
	source /etc/profile



在Master上启动Hbase(保证Hadoop启动的情况下)

	start-hbase.sh

	#停止
	stop-hbase.sh


##验证

执行

	hbase shell