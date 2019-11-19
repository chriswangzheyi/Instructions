# Docker 搭建kafa集群

### 参考资料

	https://www.cnblogs.com/jay763190097/p/10292227.html

## 搭建步骤


### 创建网络

	docker network create -d bridge kafka_network

### 编写docker-compose

vim /root//kafka_test/docker-compose.yml


	version: '2'
	services:
	  zookeeper:
	    image: wurstmeister/zookeeper   ## 镜像
		container_name: zoo1
	    ports:
	      - "2181:2181"                 ## 对外暴露的端口号
	  kafka1:
	    image: wurstmeister/kafka       ## 镜像
	    container_name: kafka1
	    ports:
	      - "9092:9092"
	    environment:
	      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://47.112.142.231:9092          ## 修改:宿主机IP
	      KAFKA_ZOOKEEPER_CONNECT: 47.112.142.231:2181                         ## 卡夫卡运行是基于zookeeper的
	      KAFKA_BROKER_ID: 0
	      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
		  
	
	  kafka2:
	    image: wurstmeister/kafka       ## 镜像
	    container_name: kafka2
	    ports:
	      - "9093:9092"
	    environment:
	      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://47.112.142.231:9093          ## 修改:宿主机IP
	      KAFKA_ZOOKEEPER_CONNECT: 47.112.142.231:2181                         ## 卡夫卡运行是基于zookeeper的
	      KAFKA_BROKER_ID: 1
	      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
	
	  kafka3:
	    image: wurstmeister/kafka       ## 镜像
	    container_name: kafka3
	    ports:
	      - "9094:9092"
	    environment:
	      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://47.112.142.231:9094          ## 修改:宿主机IP
	      KAFKA_ZOOKEEPER_CONNECT: 47.112.142.231:2181                         ## 卡夫卡运行是基于zookeeper的
	      KAFKA_BROKER_ID: 2
		  KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092

	  kafka-manager:
	    image: sheepkiller/kafka-manager       ## 镜像
	    container_name: kafka-manager
	    restart: always
	    ports:
	      - "9001:9000"
	    links:
          - kafka1
          - kafka2
          - kafka3
        external_links:   # 连接本compose文件以外的container
          - zoo1
	    environment:
	      ZK_HOSTS: zoo1:2181
	      KAFKA_BROKERS: kafka1:9092,kafka2:9093,kafka3:9094
	      APPLICATION_SECRET: letmein
	      KM_ARGS: -Djava.net.preferIPv4Stack=true          
	networks:
	  default:
	    external:   # 使用已创建的网络
	      name: kafka_network	  


###启动

	docker-compose up -d

## 验证

### Kafka-manager 查看集群状态

访问9001端口：

	http://47.112.142.231:9001

配置：

第一步： add cluster

![](../Images/1.png)

第二步： 填写zk等其他配置

![](../Images/2.png)

第三步：查看集群

![](../Images/3.png)


