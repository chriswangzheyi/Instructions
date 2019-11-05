# Flume 安装流程


##集群搭建

###解压

	tar -xzvf apache-flume-1.9.0-bin.tar.gz

### 修改配置文件

	cp /root/apache-flume-1.9.0-bin/conf/flume-env.sh.template /root/apache-flume-1.9.0-bin/conf/flume-env.sh

	vi /root/apache-flume-1.9.0-bin/conf/flume-env.sh

	# 增加
	export JAVA_HOME=/root/jdk1.8.0_221

### 修改环境变量

	vi /etc/profile
	
	#插入
	export FLUME_HOME=/root/apache-flume-1.9.0-bin
	export FLUME_CONF=$FLUME_HOME/conf	
	export PATH=$PATH:$FLUME_HOME/bin

	#刷新配置
	source /etc/profile



### 验证
	
	flume-ng version


### 将安装包复制到其他节点

	scp -r /root/apache-flume-1.9.0-bin Slave001:/root
	
	scp -r /root/apache-flume-1.9.0-bin Slave002:/root

	scp -r /root/apache-flume-1.9.0-bin Slave003:/root

### 在Slave001, Slave002, Slave003中更新环境变量：

	vi /etc/profile
	
	#插入
	export FLUME_HOME=/root/apache-flume-1.9.0-bin
	export FLUME_CONF=$FLUME_HOME/conf	
	export PATH=$PATH:$FLUME_HOME/bin

	#刷新配置
	source /etc/profile	



## 配置

在Slave中：

	vi /root/apache-flume-1.9.0-bin/conf/netcat.conf


插入

	# Name the components on this agent 定义Agent
	a1.sources = r1
	a1.sinks = k1
	a1.channels = c1
	
	# Describe/configure the source 描述source
	a1.sources.r1.type = netcat
	a1.sources.r1.bind = Slave001
	a1.sources.r1.port = 44444
	
	# Describe the sink 描述sink
	a1.sinks.k1.type = logger
	
	# Use a channel which buffers events in memory 使用缓冲池
	a1.channels.c1.type = memory
	a1.channels.c1.capacity = 1000
	a1.channels.c1.transactionCapacity = 100
	
	# Bind the source and sink to the channel 绑定
	a1.sources.r1.channels = c1
	a1.sinks.k1.channel = c1

###启动

	flume-ng agent --conf conf/ --name a1 --conf-file /root/apache-flume-1.9.0-bin/conf/netcat.conf -Dflume.root.logger==INFO,console

###验证

使用telnet工具向本机的44444端口发送内容

安装telnet

	
	yum install -y telnet.x86_64
	yum install -y telnet-server.x86_64

使用命令发送内容

 	telnet Slave001 44444


发送方：

![](../Images/2.png)

接收方：

![](../Images/3.png)


#将日志写入HDFS

### 新建文件夹

	mkdir /root/apache-flume-1.9.0-bin/example

### 新建配置文件

	vi /root/apache-flume-1.9.0-bin/example/log-hdfs.conf

插入

	# Name the components on this agent
	a1.sources = r1
	a1.sinks = k1
	a1.channels = c1
	 
	#exec 指的是命令
	# Describe/configure the source
	a1.sources.r1.type = exec
	#F根据文件名追中, f根据文件的nodeid追中
	a1.sources.r1.command = tail -F /root/hadoop-3.2.1/testdata/testflume.log
	a1.sources.r1.channels = c1
	 
	# Describe the sink
	#下沉目标
	a1.sinks.k1.type = hdfs
	a1.sinks.k1.channel = c1
	#指定目录, flum帮做目的替换
	a1.sinks.k1.hdfs.path = /flume/events/%y-%m-%d/%H%M/
	#文件的命名, 前缀
	a1.sinks.k1.hdfs.filePrefix = events-
	 
	#10 分钟就改目录（创建目录）， （这些参数影响/flume/events/%y-%m-%d/%H%M/）
	a1.sinks.k1.hdfs.round = true
	a1.sinks.k1.hdfs.roundValue = 10
	a1.sinks.k1.hdfs.roundUnit = minute
	#目录里面有文件
	#------start----两个条件，只要符合其中一个就满足---
	#文件滚动之前的等待时间(秒)
	a1.sinks.k1.hdfs.rollInterval = 3
	#文件滚动的大小限制(bytes)
	a1.sinks.k1.hdfs.rollSize = 500
	#写入多少个event数据后滚动文件(事件个数)
	a1.sinks.k1.hdfs.rollCount = 20
	#-------end-----
	 
	#5个事件就往里面写入
	a1.sinks.k1.hdfs.batchSize = 5
	 
	#用本地时间格式化目录
	a1.sinks.k1.hdfs.useLocalTimeStamp = true
	 
	#下沉后, 生成的文件类型，默认是Sequencefile，可用DataStream，则为普通文本
	a1.sinks.k1.hdfs.fileType = DataStream
	 
	# Use a channel which buffers events in memory
	a1.channels.c1.type = memory
	a1.channels.c1.capacity = 1000
	a1.channels.c1.transactionCapacity = 100
	 
	# Bind the source and sink to the channel
	a1.sources.r1.channels = c1
	a1.sinks.k1.channel = c1



### 新建文件夹

用于存Log

	mkdir /root/hadoop-3.2.1/testdata


	cd /root/hadoop-3.2.1/testdata

	#执行命令
	while true; do echo "hello testing do" >> testflume.log ; sleep 0.5; done

	#另外开一个窗口
	tail -f testflume.log 

	可以看到testflume.log文件内容不断增加


### 启动Hadoop
	
	start-all.sh
	hdfs haadmin -transitionToActive --forcemanual  nn1


###替换	guava

如果报错：

	java.lang.NoSuchMethodError: com.google.common.base.Preconditions.checkArgument(ZLjava/lang/String;Ljava/lang/Object;)V

将gguava-23.6-jre.jar上传到lib文件夹中，删除原来的guava jar包


### 启动Flume

	flume-ng agent --conf conf/ --name a1 --conf-file /root/apache-flume-1.9.0-bin/example/log-hdfs.conf -Dflume.root.logger==INFO,console


### 验证


![](../Images/4.png)

在hadoop管理页面中：

![](../Images/5.png)




参考资料

> https://blog.csdn.net/weixin_36058293/article/details/81189891
	