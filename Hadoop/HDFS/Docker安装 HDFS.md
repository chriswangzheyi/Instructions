# Docker安装 HDFS

参考资料： https://www.cnblogs.com/fanshudada/p/11781325.html

## 安装步骤

### 拉取镜像

	docker pull singularities/hadoop

### docker-compose

	vi docker-compose.yml

插入：

	version: "2"
	
	services:
	  namenode:
	    image: singularities/hadoop
	    command: start-hadoop namenode
	    hostname: namenode
	    environment:
	      HDFS_USER: hdfsuser
	    ports:
	      - "8020:8020"
	      - "14000:14000"
	      - "50070:50070"
	      - "50075:50075"
	      - "10020:10020"
	      - "13562:13562"
	      - "19888:19888"
	  datanode:
	    image: singularities/hadoop
	    command: start-hadoop datanode namenode
	    environment:
	      HDFS_USER: hdfsuser
	    links:
	      - namenode



### 启动（单点）：

	docker-compose up -d

这种启动方式是1个dataNode, 一个nameNode

### 扩容为集群：

	docker-compose --scale datanode=3 

这种启动方式新增3个dataNode


### 验证

	http://47.112.142.231:50070/dfshealth.html#tab-datanode

![](../Images/1.png)



## 容器内操作

### 进入容器

	docker exec -it {容器id} /bin/bash 

### 基本操作

#### 创建目录

	hadoop fs -mkdir /hdfs #在根目录下创建hdfs文件夹

#### 查看目录

	hadoop fs -ls / #列出跟目录下的文件列表

#### 多级创建目录

	hadoop fs -mkdir -p /hdfs/d1/d2

#### 级联列出目录

	hadoop fs -ls -R /

#### 上传本地文件到HDFS

	echo "hello hdfs" >>local.txt
	hadoop fs -put local.txt /hdfs/d1/d2


#### 查看HDFS中文件的内容

	hadoop fs -cat /hdfs/d1/d2/local.txt


#### 下载hdfs上文件的内容

	hadoop fs -get /hdfs/d1/d2/local.txt

#### 删除hdfs文件

	hadoop fs -rm /hdfs/d1/d2/local.txt

#### 删除hdfs中目录

	hadoop fs -rmdir /hdfs/d1/d2