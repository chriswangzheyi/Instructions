# Docker安装spark

https://github.com/big-data-europe/docker-spark


## 安装

	vim docker-compose.yml

插入：

	version: '3'
	services:
	  spark-master:
	    image: bde2020/spark-master:3.1.1-hadoop3.2
	    container_name: spark-master
	    ports:
	      - "8080:8080"
	      - "7077:7077"
	    environment:
	      - INIT_DAEMON_STEP=setup_spark
	  spark-worker-1:
	    image: bde2020/spark-worker:3.1.1-hadoop3.2
	    container_name: spark-worker-1
	    depends_on:
	      - spark-master
	    ports:
	      - "8081:8081"
	    environment:
	      - "SPARK_MASTER=spark://spark-master:7077"
	  spark-worker-2:
	    image: bde2020/spark-worker:3.1.1-hadoop3.2
	    container_name: spark-worker-2
	    depends_on:
	      - spark-master
	    ports:
	      - "8082:8081"
	    environment:
	      - "SPARK_MASTER=spark://spark-master:7077"

## 启动

	docker-compose up
	      
##  验证

	http://47.112.142.231:8080/

