#02安装hadoop


## 准备工作

在192.168.195.150上操作

### 关闭防火墙

	systemctl stop firewalld.service

 	systemctl disable firewalld.service

### 修改Hostname

	vim /etc/hostname

编译为

	master

### 修改host

	vim /etc/hosts

配置

	192.168.195.150 master
	192.168.195.151 slave


### 创建免密

	ssh-keygen -t rsa 

一路回车

	ssh-copy-id master

分别输入yes 和 密码


## 安装hadoop
 
	tar -zxf hadoop-2.7.5.tar.gz
	mv hadoop-2.7.5 /usr/local/hadoop
	cd /usr/local/hadoop/etc/hadoop

### 修改配置文件

vim core-site.xml 

插入

    <configuration>
		<property>
		  <name>fs.default.name</name>
		  <value>hdfs://master:9000</value>
		</property>
		<property>
		  <name>hadoop.tmp.dir</name>
		  <value>/usr/local/hadoop/tmp</value>
		</property>
		<property>
		    <name>hadoop.proxyuser.root.hosts</name>
		    <value>*</value>
		</property>
		<property>
		    <name>hadoop.proxyuser.root.groups</name>
		    <value>*</value>
		</property>
    </configuration>


vim hdfs-site.xml


	<configuration>
		<property>
		  <name>dfs.replication</name>
		  <value>1</value>
		</property>
		 <property>
		        <name>dfs.permissions</name>
		        <value>false</value>
		    </property>
	</configuration>


cp mapred-site.xml.template mapred-site.xml

vim mapred-site.xml

	<configuration>
		<property>
		  <name>mapred.job.tracker</name>
		  <value>master:9001</value>
		</property>
	</configuration>


vim slaves

	master
	slave

vim hadoop-env.sh

修改

	export JAVA_HOME=/usr/local/jdk

### 修改配置文件

vim /etc/profile

修改为：

	export JAVA_HOME=/usr/local/jdk
	export ZK_HOME=/usr/local/zk
	export KAFKA_HOME=/usr/local/kafka
	export HADOOP_HOME=/usr/local/hadoop
	export PATH=$PATH:$JAVA_HOME/bin:$ZK_HOME/bin:$KAFKA_HOME/bin:$HADOOP_HOME/bin
 

source /etc/profile


##安装slave

在192.168.195.151上操作

### 关闭防火墙

	systemctl stop firewalld.service

 	systemctl disable firewalld.service

### 修改Hostname

	vim /etc/hostname

编译为

	slave

### 修改host

	vim /etc/hosts

配置

	192.168.195.150 master
	192.168.195.151 slave


### 创建免密

	ssh-keygen -t rsa 

一路回车

	ssh-copy-id slave

分别输入yes 和 密码

## master slave 互相免密

在master服务器上做

	ssh-copy-id slave

在slave服务器上做

	ssh-copy-id master

## 复制安装包和配置文件

在192.168.195.150上操作


	scp -r /usr/local/jdk slave:/usr/local/

	scp -r /usr/local/hadoop/ slave:/usr/local/

	scp -r /etc/profile slave:/etc/

	scp -r /etc/hosts slave:/etc/


## 让配置生效

在192.168.195.151上操作

	source /etc/profile

## 格式化namenode

在192.168.195.150上操作

	hadoop namenode -format


## 启动Hadoop

	cd /usr/local/hadoop/sbin
	./start-dfs.sh


成功后，输入jps查看

### master

	[root@localhost sbin]# jps
	2161 Kafka
	13203 SecondaryNameNode
	1780 QuorumPeerMain
	13046 DataNode
	13318 Jps
	12922 NameNode

### slave

	12472 DataNode
	12536 Jps


## 验证

访问：

	http://192.168.195.150:50070/dfshealth.html#tab-overview

