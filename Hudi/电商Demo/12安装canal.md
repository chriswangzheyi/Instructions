# 12安装canal


## 解压

	mkdir /usr/local/canal
	cp canal.deployer-1.1.4.tar.gz  /usr/local/canal/
	cd /usr/local/canal
	tar -zxvf canal.deployer-1.1.4.tar.gz

## 修改配置文件

	cd /usr/local/canal/conf
	vim canal.properties 

最底部插入

	# 默认值tcp，这里改为投递到Kafka
	canal.serverMode = kafka
	# Kafka bootstrap.servers，可以不用写上全部的brokers
	canal.mq.servers = 192.168.195.150:9092
	# 投递失败的重试次数，默认0，改为2
	canal.mq.retries = 2
	# Kafka batch.size，即producer一个微批次的大小，默认16K，这里加倍
	canal.mq.batchSize = 32768
	# Kafka max.request.size，即一个请求的最大大小，默认1M，这里也加倍
	canal.mq.maxRequestSize = 2097152
	# Kafka linger.ms，即sender线程在检查微批次是否就绪时的超时，默认0ms，这里改为200ms
	# 满足batch.size和linger.ms其中之一，就会发送消息
	canal.mq.lingerMs = 200
	# Kafka buffer.memory，缓存大小，默认32M
	canal.mq.bufferMemory = 33554432
	# 获取binlog数据的批次大小，默认50
	canal.mq.canalBatchSize = 50
	# 获取binlog数据的超时时间，默认200ms
	canal.mq.canalGetTimeout = 200
	# 是否将binlog转为JSON格式。如果为false，就是原生Protobuf格式
	canal.mq.flatMessage = true
	# 压缩类型，官方文档没有描述
	canal.mq.compressionType = none
	# Kafka acks，默认all，表示分区leader会等所有follower同步完才给producer发送ack
	# 0表示不等待ack，1表示leader写入完毕之后直接ack
	canal.mq.acks = all
	# Kafka消息投递是否使用事务
	# 主要针对flatMessage的异步发送和动态多topic消息投递进行事务控制来保持和Canal binlog位置的一致性
	# flatMessage模式下建议开启
	canal.mq.transaction = true

修改

	canal.instance.tsdb.dbPassword = 1qa2ws#ED


### 修改example

	cd /usr/local/canal/conf/example
	vim instance.properties

修改

	canal.instance.master.address=192.168.195.150:3306
	canal.instance.dbPassword=1qa2ws#ED

插入

	canal.instance.defaultDatabaseName=canaltest

## 创造mysql用户并赋权

	CREATE USER canal IDENTIFIED BY '1qa2ws#ED';
	GRANT SELECT, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'canal'@'%';
	ALTER USER 'canal'@'%' IDENTIFIED WITH mysql_native_password BY '1qa2ws#ED';
	grant all privileges on *.* to 'canal'@'%'
	FLUSH PRIVILEGES;


## 查看binlog 开启情况

进入mysql

	mysql> show variables like '%log_bin%';
	+---------------------------------+-----------------------------+
	| Variable_name                   | Value                       |
	+---------------------------------+-----------------------------+
	| log_bin                         | ON                          |
	| log_bin_basename                | /var/lib/mysql/binlog       |
	| log_bin_index                   | /var/lib/mysql/binlog.index |
	| log_bin_trust_function_creators | OFF                         |
	| log_bin_use_v1_row_events       | OFF                         |
	| sql_log_bin                     | ON                          |
	+---------------------------------+-----------------------------+



## 验证

### 启动kafka消费者

	./kafka-console-consumer.sh --bootstrap-server 192.168.195.150:9092 --topic example

### 启动canal

	cd /usr/local/canal/bin
	startup.sh 

### 收到信息

	{"data":null,"database":"","es":1615009527000,"id":1,"isDdl":false,"mysqlType":null,"old":null,"pkNames":null,"sql":"ALTER USER 'canal'@'%' IDENTIFIED WITH 'mysql_native_password' AS '*51F35DAC4C082CB3DDCE8CE8F6F6146FFC5DFDC5'","sqlType":null,"table":"","ts":1615009963408,"type":"QUERY"}
	{"data":[{"id":"12","info":"111"}],"database":"datalake","es":1615009992000,"id":2,"isDdl":false,"mysqlType":{"id":"int","info":"varchar(10)"},"old":[{"id":"1"}],"pkNames":["id"],"sql":"","sqlType":{"id":4,"info":12},"table":"test","ts":1615009992587,"type":"UPDATE"}

