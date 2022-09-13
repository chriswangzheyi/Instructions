# Flume 集群安装

## 准备机器

* 192.168.2.111
* 192.168.2.112
* 192.168.2.113

## 安装步骤

上传安装包到三个服务器

	tar -xzvf  apache-flume-1.10.1-bin.tar.gz
	mv apache-flume-1.10.1-bin flume
	
	cd /home/zheyi/flume/conf
	cp flume-env.sh.template flume-env.sh
	vim flume-env.sh
	
修改

	export JAVA_HOME=/home/zheyi/jdk1.8.0_341
	export JAVA_OPTS="-Xms100m -Xmx2000m -Dcom.sun.management.jmxremote"