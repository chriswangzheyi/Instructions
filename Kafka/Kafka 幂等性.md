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


## 