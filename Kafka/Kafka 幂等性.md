# Kafka 幂等性




## 方法一：application.yml中增加enable.idempotence: true

	server:
	  port: 8888
	spring:
	  kafka:
	    producer:
	      bootstrap-servers: 47.112.142.231:9092 #服务器ip+端口
	      properties:
	        enable.idempotence: true


## 方法二：为producer增加pid

为了实现Producer的幂等性，Kafka引入了Producer ID（即PID）和Sequence Number。在后续插入redis或者数据库的时候采用唯一索引可以去重。