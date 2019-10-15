#Hive 安装



##安装mysql


yum -y install wget


直接下载了安装用的Yum Repository，大概25KB的样子，然后就可以直接yum安装了

	wget http://dev.mysql.com/get/mysql57-community-release-el7-11.noarch.rpm

通过yum安装

	yum -y install mysql57-community-release-el7-11.noarch.rpm
 
安装Mysql服务器

	yum install mysql-community-server

启动Mysql并保持开机启动

	systemctl start mysqld
	systemctl enable mysqld
	systemctl daemon-reload

查看mysql初始密码

	grep "password" /var/log/mysqld.log  

	示例：得到密码：9fqTklsnxB_( 

进入mysql

	mysql -uroot -p

修改密码(f非必须)

	ALTER USER 'root'@'localhost' IDENTIFIED BY 'Abcd=1234';

授权远程访问

	GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY 'Abcd=1234' WITH GRANT OPTION;




##安装Hive步骤

将apache-hive-2.3.6-bin.tar.gz安装包上传至服务器 或者通过


	wget https://mirrors.tuna.tsinghua.edu.cn/apache/hive/hive-2.3.6/apache-hive-2.3.6-bin.tar.gz

解压

	tar -xzvf apache-hive-2.3.6-bin.tar.gz

vi /etc/profile

	export HIVE_HOME=/root/apache-hive-2.3.6-bin
	export PATH=$PATH:$HIVE_HOME/bin
	export CLASSPATH=$CLASSPATH:$HIVE_HOME/lib

使得配置生效

	source /etc/profile




cd /root/apache-hive-2.3.6-bin/conf
	
	cp hive-default.xml.template hive-site.xml
	cp hive-env.sh.template hive-env.sh
	cp hive-log4j2.properties.template hive-log4j2.properties
	cp hive-exec-log4j2.properties.template hive-exec-log4j2.properties


修改配置文件

vi /root/apache-hive-2.3.6-bin/conf/hive-env.sh


	JAVA_HOME=/root/jdk1.8.0_221    # 你的java文件路径
	HADOOP_HOME=/root/hadoop-3.2.1    # 你的hadoop路径
	export HIVE_CONF_DIR=/root/apache-hive-2.3.6-bin/conf

vi /root/apache-hive-2.3.6-bin/conf/hive-site.xml


修改以下四个参数（建议下载到本地修改后覆盖原文件）

	<property>
	<name>javax.jdo.option.ConnectionURL</name>
	<value>jdbc:mysql://localhost:3306/hive</value>
	<description>JDBC connect string for a JDBC metastore</description>
	</property>
	 
	<property>
	<name>javax.jdo.option.ConnectionDriverName</name>
	<value>com.mysql.jdbc.Driver</value>
	<description>Driver class name for a JDBC metastore</description>
	</property>
	 
	<property>
	<name>javax.jdo.option.ConnectionUserName</name>
	<value>root</value>#数据库用户名
	<description>Username to use against metastore database</description>
	</property>
	 
	<property>
	<name>javax.jdo.option.ConnectionPassword</name>
	<value>Abcd=1234</value>#数据库密码
	<description>password to use against metastore database</description>
	</property>
	
	#以下几个不修改会出错

	<property>
	<name>hive.exec.local.scratchdir</name>
	<value>/root/apache-hive-2.3.6-bin/<</value>
	<description>Local scratch space for Hive jobs</description>
	</property>
	 
	<property>
	<name>hive.downloaded.resources.dir</name>
	<value>/root/apache-hive-2.3.6-bin/hive-downloaded-addDir/</value>#自定义目录
	<description>Temporary local directory for added resources in the remote file system.</description>
	</property>
	 
	<property>
	<name>hive.querylog.location</name>
	<value>/root/apache-hive-2.3.6-bin/querylog-location-addDir/</value>#自定义目录
	<description>Location of Hive run time structured log file</description>
	</property>
	 
	<property>
	<name>hive.server2.logging.operation.log.location</name>
	<value>/root/apache-hive-2.3.6-bin/hive-logging-operation-log-addDir/</value>#自定义目录
	<description>Top level directory where operation logs are stored if logging functionality is enabled</description>
	</property>


## 替换guava文件 （重要）


比较hadoop 和 hive guava文件的版本，删除低版本的，替换为高版本

路径分别为：

	/root/hadoop-3.2.1/share/hadoop/common/lib


	/root/apache-hive-2.3.6-bin/lib

	
## 验证


在启动hadoop集群后

	hive