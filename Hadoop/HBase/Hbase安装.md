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
	export HBASE_HEAPSIZE=1G


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
	        <value>/root/zookeeper-3.4.14/temp/zookeeper</value>
	</property>	
	<property>
		<!-- 允许节点时间偏差值 -->
		<name>hbase.master.maxclockskew</name>
		<value>180000</value>
	</property>
	<!-- ZooKeeper 会话超时。Hbase 把这个值传递给 zk 集群，向它推荐一个会话的最大超时时间 -->
    <property>
        <name>zookeeper.session.timeout</name>
        <value>120000</value>
    </property>
	<!-- 当 regionserver 遇到 ZooKeeper session expired ， regionserver 将选择 restart 而不是 abort -->
    <property>
        <name>hbase.regionserver.restart.on.zk.expire</name>
        <value>true</value>
    </property>
	<property>
		<name>hbase.unsafe.stream.capability.enforce</name>
		<value>false</value>
	</property>


其中：

	<property>
		<name>hbase.unsafe.stream.capability.enforce</name>
		<value>false</value>
	</property>

这一个配置如果不配会导致HMaster自动关闭



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


将Zookeeper节点替换为同一版本

	cp /root/zookeeper-3.4.14.jar /root/hbase-2.2.1/lib/
	rm  /root/hbase-2.2.1/lib/zookeeper-3.4.10.jar

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



**在Master上启动Hbase**
	
步骤一：在Slave001, Slave002, Slave003中启动zookeeper

	
步骤二： 启动Hadoop
	
	start-all.sh

	hdfs haadmin -transitionToActive --forcemanual  nn1
	
	hdfs dfsadmin -safemode leave

验证hadoop状态

	http://192.168.195.128:50070/dfshealth.html#tab-overview


##删除Hadoop 损坏的文件(非必须)

	hadoop fsck -delete /

	
步骤三： 启动Hbases

	start-hbase.sh

	#停止
	stop-hbase.sh

	#在Slave001, Slave002, Slave003中
	hbase-daemon.sh start regionserver

	#在Master001, Master002
	hbase-daemon.sh start master




##验证


访问16010端口

	http://192.168.195.128:16010


或者执行

	hbase shell



