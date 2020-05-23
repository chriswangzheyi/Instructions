# 单节点搭建kafka

## 安装步骤

### 解压

	tar -zxvf kafka_2.13-2.5.0.tgz


### 单节点 - 单代理配置

kafka依赖zookeeper，所以需要先启动kafka自带的zookeeper服务器

	cd /root/kafka_2.13-2.5.0/bin
	
启动zk：

	./zookeeper-server-start.sh ../config/zookeeper.properties
	或后台启动
	nohup ./zookeeper-server-start.sh ../config/zookeeper.properties >> zookeeper.out &


### 启动kafka

	启动Kafka server
	./kafka-server-start.sh ../config/server.properties
	或后台启动
	nohup ./kafka-server-start.sh ../config/server.properties >> kafka.out &


### 验证是否启动

	jps


## 验证

创建一个topic:

	./kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test

响应结果:Created topic test

创建成功，输入

	./kafka-topics.sh --list --zookeeper localhost:2181 

进行查看

结果：输出test

发送消息：

	./kafka-console-producer.sh --broker-list localhost:9092 --topic test

之后输入想要发送的消息

接收消息：

	./kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic test --from-beginning

即时响应发送的消息


