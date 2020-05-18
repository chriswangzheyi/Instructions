# Zookeeper 安装 (单节点)

## 安装步骤

### 解压

	tar -zxvf /root/apache-zookeeper-3.6.1-bin.tar.gz -C /root

### 修改配置文件

	cd /root/apache-zookeeper-3.6.1-bin/conf/
	cp zoo_sample.cfg zoo.cfg 

	vim zoo.cfg

修改

	dataDir=/root/apache-zookeeper-3.6.1-bin/zkData

### 创建文件夹

	mkdir /root/apache-zookeeper-3.6.1-bin/zkData

## 验证

### 启动



	cd /root/apache-zookeeper-3.6.1-bin/bin
	./zkServer.sh start

### 查看是否启动

	jps

显示

	4008 Jps
	3961 QuorumPeerMain

### 查看状态
	cd /root/apache-zookeeper-3.6.1-bin/bin
	./zkServer.sh status



#### 进入客户端

	./zkCli.sh
