# Docker 搭建flink


## 拉取镜像

	docker pull flink

## docker-compose.yml

	version: "2.1"
	services:
	  jobmanager:
	    image: flink
	    expose:
	      - "6123"
	    ports:
	      - "8081:8081"
	    command: jobmanager
	    environment:
	      - JOB_MANAGER_RPC_ADDRESS=jobmanager
	 
	  taskmanager:
	    image: flink
	    expose:
	      - "6121"
	      - "6122"
	    depends_on:
	      - jobmanager
	    command: taskmanager
	    links:
	      - "jobmanager:jobmanager"
	    environment:
	      - JOB_MANAGER_RPC_ADDRESS=jobmanager
	      

## 启动

	docker-compose up -d

## 访问

浏览器打开 

	http://47.112.142.231:8081/

可以看到dashboard