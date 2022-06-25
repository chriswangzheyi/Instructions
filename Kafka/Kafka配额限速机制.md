# Kafka配额限速机制

## Kafka配额限速机制
生产者和消费者以极高的速度生产/消费大量数据或产生请求，从而占用broker上的全部资源，造成网络IO饱和。有了配额（Quotas）就可以避免这些问题。Kafka支持配额管理，从而可以对Producer和Consumer的produce&fetch操作进行流量限制，防止个别业务压爆服务器。

## 限制producer端的速率

为所有client id设置默认值，以下为所有producer程序设置其TPS不超过1MB/s，即1048576/s，命令如下：

	bin/kafka-configs.sh --zookeeper node1:2181 --alter --add-config 'producer_byte_rate=1048576' --entity-type clients --entity-default
	
运行基准测试，观察生产消息的速率

	bin/kafka-producer-perf-test.sh --topic test --num-records 50000 --throughput -1 --record-size 1000 --producer-props bootstrap.servers=node1:9092,node2:9092,node3:9092 acks=1
	
## 限制consumer端的速率

为指定的topic进行限速，以下为所有consumer程序设置topic速率不超过1MB/s，即1048576/s。命令如下：

	bin/kafka-configs.sh --zookeeper node1:2181 --alter --add-config 'consumer_byte_rate=1048576' --entity-type clients --entity-default
	
运行基准测试，观察消息消费的速率

	bin/kafka-consumer-perf-test.sh --broker-list node1:9092,node2:9092,node3:9092 --topic test --fetch-size 1048576 --messages 50000
	
## 取消kafka的Quota配置

	bin/kafka-configs.sh --zookeeper node1:2181 --alter --delete-config 'producer_byte_rate' --entity-type clients --entity-default
	
	bin/kafka-configs.sh --zookeeper node1:2181 --alter --delete-config 'consumer_byte_rate' --entity-type clients --entity-default