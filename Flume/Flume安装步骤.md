# Flume安装步骤

---

## 前期准备

安装java:

	yum -y install java-1.8.0-openjdk*



## 下载文件

下载网站：

http://www.apache.org/dyn/closer.lua/flume/1.9.0/apache-flume-1.9.0-bin.tar.gz

## 解压

	tar -xzvf apache-flume-1.9.0-bin.tar.gz
	mv apache-flume-1.9.0-bin flume

## 修改配置文件

	vi /etc/profile

插入

	export FLUME_HOME=/root/flumeDemo/flume
	export PATH=$PATH:$FLUME_HOME/bin
	
刷新

	source /etc/profile

验证是否成功

	cd /root/flumeDemo/flume/bin
	flume-ng version

