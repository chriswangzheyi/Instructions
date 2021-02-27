# 05 安装spark和scala

## 解压

	tar -zxvf spark-2.4.7-bin-hadoop2.7.tgz
	tar -zxvf scala-2.13.5.tgz

 	mv spark-2.4.7-bin-hadoop2.7 /usr/local/spark
	mv scala-2.13.5 /usr/local/scala


## 配置

vim /etc/profile

	export SCALA_HOME=/usr/local/scala
	export SPARK_HOME=/usr/local/spark
	export PATH=$PATH:$SCALA_HOME/bin:$SPARK_HOME/bin


cd /usr/local/spark/conf

	cp spark-env.sh.template spark-env.sh

插入

	# 配置JAVA_HOME，一般来说，不配置也可以，但是可能会出现问题，还是配上吧
	export JAVA_HOME=/usr/local/jdk
	# 一般来说，spark任务有很大可能性需要去HDFS上读取文件，所以配置上
	# 如果说你的spark就读取本地文件，也不需要yarn管理，不用配
	export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
	# 设置Master的主机名
	export SPARK_MASTER_HOST=master
	# 提交Application的端口，默认就是这个，万一要改呢，改这里
	export SPARK_MASTER_PORT=7077
	# 每一个Worker最多可以使用的cpu core的个数，我虚拟机就一个...
	# 真实服务器如果有32个，你可以设置为32个
	export SPARK_WORKER_CORES=1
	# 每一个Worker最多可以使用的内存，我的虚拟机就2g
	# 真实服务器如果有128G，你可以设置为100G
	export SPARK_WORKER_MEMORY=1g
	#设置pid存储位置
	export SPARK_PID_DIR=/usr/local/spark/pids


mkdir /usr/local/spark/pids

cd /usr/local/spark/sbin

	mv start-all.sh start-spark-all.sh
	mv stop-all.sh stop-spark-all.sh


### 启动Spark集群

	start-spark-all.sh

### 停止Spark集群

	stop-spark-all.sh


查看管理页面

	http://192.168.195.150:8080
