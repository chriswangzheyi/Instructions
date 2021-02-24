# 01安装Kafka和zk

## 安装

在192.168.195.150上操作

### 解压文件

	tar -zxvf jdk-8u181-linux-x64.gz 
	
	tar -zxvf kafka_2.11-2.1.0.tgz 
	
	tar -zxvf zookeeper-3.4.5.tar.gz

### 解压文件

	mv jdk1.8.0_181/ /usr/local/jdk

	mv zookeeper-3.4.5 /usr/local/zk

	mv kafka_2.11-2.1.0  /usr/local/kafka


## 配置组件

### 配置zk

	cd /usr/local/zk/conf
	cp zoo_sample.cfg zoo.cfg
	vim zoo.cfg

修改

	dataDir=/usr/local/zk/data

### 配置kfaka

	cd /usr/local/kafka/config
	vim server.properties 

修改

	log.dirs=/usr/local/kafka/data

## 配置环境变量

vim /etc/profile

	export JAVA_HOME=/usr/local/jdk
	export ZK_HOME=/usr/local/zk
	export KAFKA_HOME=/usr/local/kafka
	export PATH=$PATH:$JAVA_HOME/bin:$ZK_HOME/bin:$KAFKA_HOME/bin

## 启动

### 启动zk

	cd /usr/local/zk/bin
	./zkServer.sh start

### 启动Kafka

	cd /usr/local/kafka/bin/
	./kafka-server-start.sh ../config/server.properties 

	或后台启动
	nohup ./kafka-server-start.sh ../config/server.properties >> kafka.out &

