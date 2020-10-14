# 使用Ambari 部署集群 （HDP方式）

## 前置条件

准备三台虚拟机，ip分别是：192.168.3.21， 192.168.3.22，192.168.3.23

192.168.3.21 需要2核8G或以上

## 对每台服务器进行操作

### 关闭防火墙

	systemctl stop firewalld.service
	systemctl disable firewalld.service

修改hosts
	
	vi /etc/hosts
	
插入

	192.168.3.21 hdp21
	192.168.3.22 hdp22
	192.168.3.23 hdp23

### 安装JDK

	yum install java-1.8.0-openjdk* -y;


jdk 被安装到

	/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.262.b10-0.el7_8.x86_64/

vi /etc/profile

	export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.252.b09-2.el7_8.x86_64
	export JRE_HOME=$JAVA_HOME/jre  
	export PATH=$PATH:$JAVA_HOME/bin:$JRE_HOME/bin
	export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar:$JRE_HOME/lib
	
	
### 设置免密登录

#### 测试是否可以免密登录

	ssh localhost
需要输入密码则表示不免密

#### 设置免密登录
	
	$ cd /root/.ssh
	
	$ ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
	$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
	$ chmod 0600 ~/.ssh/authorized_keys
	
#### 验证

	ssh localhost
	
### 安装NTP服务

Network Time Protocol(NTP)是用来使计算机时间同步化的一种协议

	yum install ntp
	systemctl is-enabled ntpd。# 查看是否启动
	systemctl enable ntpd
	systemctl start ntpd


## 对92.168.3.21进行操作

作为主节点

### 设置主机名

	sudo hostnamectl set-hostname hdp21
	
### 安装Apache Http服务

	yum install httpd
	
启动：

	systemctl start httpd.service
	systemctl enable httpd.service	
	
#### 验证：

浏览器访问

	192.168.3.21
	
出现Testing 123..字样则表示成功

### 安装MariaDB

#### 添加 MariaDB yum 仓库

	vi /etc/yum.repos.d/MariaDB.repo

插入

	[mariadb]
	name = MariaDB
	baseurl = https://mirrors.ustc.edu.cn/mariadb/yum/10.2/centos7-amd64/
	gpgkey=https://mirrors.ustc.edu.cn/mariadb/yum/RPM-GPG-KEY-MariaDB
	gpgcheck=1


导入GPG key：

	rpm --import https://yum.mariadb.org/RPM-GPG-KEY-MariaDB
	
#### 安装

	yum install MariaDB-server MariaDB-client -y	
	
#### 启动

	systemctl start mariadb
	systemctl enable mariadb
	
	查看
	systemctl status mariadb
	
#### 对 MariaDB 进行安全配置

##### 安装

	mysql_secure_installation
	
	首先是设置密码，会提示先输入密码
	
	Enter current password for root (enter for none):<–初次运行直接回车
	
	设置密码
	
	Set root password? [Y/n] <– 是否设置root用户密码，输入y并回车或直接回车
	New password: <– 设置root用户的密码
	Re-enter new password: <– 再输入一次你设置的密码
	
	其他配置
	
	Remove anonymous users? [Y/n] <– 是否删除匿名用户，回车
	
	Disallow root login remotely? [Y/n] <–是否禁止root远程登录,回车（后面授权配置）
	
	Remove test database and access to it? [Y/n] <– 是否删除test数据库，回车
	
	Reload privilege tables now? [Y/n] <– 是否重新加载权限表，回车

##### 登陆数据库

	mysql -uroot -proot

##### 创建ambari数据库及用户

	create database ambari;  
	CREATE USER 'ambari'@'%'IDENTIFIED BY 'root';
	GRANT ALL PRIVILEGES ON *.* TO 'ambari'@'%';
	FLUSH PRIVILEGES;
	
##### 如果要安装Hive，再创建Hive数据库和用户 再执行下面的语句

	create database hive;  
	CREATE USER 'hive'@'%'IDENTIFIED BY 'root';
	GRANT ALL PRIVILEGES ON *.* TO 'hive'@'%';
	FLUSH PRIVILEGES;

##### 如果要安装Oozie，再创建Oozie数据库和用户 再执行下面的语句：

	create database oozie;  
	CREATE USER 'oozie'@'%'IDENTIFIED BY 'root';
	GRANT ALL PRIVILEGES ON *.* TO 'oozie'@'%';
	FLUSH PRIVILEGES;	
	
##### 	安装MySQL的JDBC驱动	

	yum install mysql-connector-java
	位置：/usr/share/java/mysql-connector-java.jar	
### 安装Ambari

#### 下载工具包

##### ambari-2.7.3.0：
	http://public-repo-1.hortonworks.com/ambari/centos7/2.x/updates/2.7.3.0/ambari-2.7.3.0-centos7.tar.gz

##### HDP-3.1.0：
	http://public-repo-1.hortonworks.com/HDP/centos7/3.x/updates/3.1.0.0/HDP-3.1.0.0-centos7-rpm.tar.gz

##### HDP-UTILS-1.1.0.22：
	http://public-repo-1.hortonworks.com/HDP-UTILS-1.1.0.22/repos/centos7/HDP-UTILS-1.1.0.22-centos7.tar.gz

##### HDP-GPL-3.1.0.0-centos7-gpl.tar.gz:

	http://public-repo-1.hortonworks.com/HDP-GPL/centos7/3.x/updates/3.1.0.0/HDP-GPL-3.1.0.0-centos7-gpl.tar.gz


#### 配置Ambari 本地源


	mkdir /var/www/html/ambari
	
	tar -zxvf ambari-2.7.3.0-centos7.tar.gz -C /var/www/html/ambari
	tar -zxvf HDP-3.1.0.0-centos7-rpm.tar.gz -C /var/www/html/ambari
	
	mkdir /var/www/html/ambari/HDP-UTILS/
	tar -zxvf HDP-UTILS-1.1.0.22-centos7.tar.gz -C /var/www/html/ambari/
	
	mkdir /var/www/html/ambari/HDP-GPL/
	tar -zxvf  HDP-GPL-3.1.0.0-centos7-gpl.tar.gz -C /var/www/html/ambari/

#### 删除目录下的html文件

	cd /var/www/html/ambari/
	find . -name *.html
	
	rm -rf 查找出来的文件

#### 安装配置Ambari Server

#### 下载repo

	wget http://public-repo-1.hortonworks.com/ambari/centos7/2.x/updates/2.7.3.0/ambari.repo
	wget http://public-repo-1.hortonworks.com/HDP/centos7/3.x/updates/3.1.0.0/hdp.repo


##### ambari.repo

	#VERSION_NUMBER=2.7.3.0-139
	[ambari-2.7.3.0]
	#json.url = http://public-repo-1.hortonworks.com/HDP/hdp_urlinfo.json
	name=ambari Version - ambari-2.7.3.0
	baseurl=http://192.168.3.21/ambari/ambari/centos7/2.7.3.0-139
	gpgcheck=1
	gpgkey=http://192.168.3.21/ambari/ambari/centos7/2.7.3.0-139/RPM-GPG-KEY/RPM-GPG-KEY-Jenkins
	enabled=1
	priority=1


#### hdp.repo

	#VERSION_NUMBER=3.1.0.0-78
	[HDP-3.1.0.0]
	name=HDP Version - HDP-3.1.0.0
	baseurl=http://192.168.3.21/ambari/HDP/centos7/3.1.0.0-78
	gpgcheck=1
	gpgkey=http://192.168.3.21/ambari/HDP/centos7/3.1.0.0-78/RPM-GPG-KEY/RPM-GPG-KEY-Jenkins
	enabled=1
	priority=1
	
	
	[HDP-UTILS-1.1.0.22]
	name=HDP-UTILS Version - HDP-UTILS-1.1.0.22
	baseurl=http://192.168.3.21/ambari/HDP-UTILS/centos7/1.1.0.22
	gpgcheck=1
	gpgkey=http://192.168.3.21/ambari/HDP-UTILS/centos7/1.1.0.22/RPM-GPG-KEY/RPM-GPG-KEY-Jenkins
	enabled=1
	priority=1


#### hdp-gpl.repo

	#VERSION_NUMBER=3.1.0.0-78
	[HDP-GPL-3.1.0.0]
	name=HDP-GPL Version - HDP-GPL-3.1.0.0
	baseurl=http://192.168.3.21/ambari/HDP-GPL/centos7/3.1.0.0-78
	gpgcheck=1
	gpgkey=http://192.168.3.21/ambari/HDP-GPL/centos7/3.1.0.0-78/RPM-GPG-KEY/RPM-GPG-KEY-Jenkins
	enabled=1
	priority=1



#### 放置repository

将ambari.repo,hdp.repo,HDP-GPL.repo 拷贝到/etc/yum.repos.d/目录下

#### 更新yum源

	sudo yum clean all
	sudo yum list update
	sudo yum makecache
	sudo yum repolist
	sudo yum upgrade openssl


#### 安装

	yum install ambari-server
	
### 配置Ambari

	ambari-server setup

步骤：

	[root@hdp21 yum.repos.d]# ambari-server setup
	Using python  /usr/bin/python
	Setup ambari-server
	Checking SELinux...
	SELinux status is 'enabled'
	SELinux mode is 'permissive'
	WARNING: SELinux is set to 'permissive' mode and temporarily disabled.
	OK to continue [y/n] (y)? y
	Customize user account for ambari-server daemon [y/n] (n)? y
	Enter user account for ambari-server daemon (root):root
	Adjusting ambari-server permissions and ownership...
	Checking firewall status...
	Checking JDK...
	[1] Oracle JDK 1.8 + Java Cryptography Extension (JCE) Policy Files 8
	[2] Custom JDK
	==============================================================================
	Enter choice (1): 2
	WARNING: JDK must be installed on all hosts and JAVA_HOME must be valid on all hosts.
	WARNING: JCE Policy files are required for configuring Kerberos security. If you plan to use Kerberos,please make sure JCE Unlimited Strength Jurisdiction Policy Files are valid on all hosts.
	Path to JAVA_HOME: /usr/lib/jvm/java-1.8.0-openjdk
	Validating JDK on Ambari Server...done.
	Check JDK version for Ambari Server...
	JDK version found: 8
	Minimum JDK version is 8 for Ambari. Skipping to setup different JDK for Ambari Server.
	Checking GPL software agreement...
	GPL License for LZO: https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
	Enable Ambari Server to download and install GPL Licensed LZO packages [y/n] (n)? y
	Completing setup...
	Configuring database...
	Enter advanced database configuration [y/n] (n)? y
	Configuring database...
	==============================================================================
	Choose one of the following options:
	[1] - PostgreSQL (Embedded)
	[2] - Oracle
	[3] - MySQL / MariaDB
	[4] - PostgreSQL
	[5] - Microsoft SQL Server (Tech Preview)
	[6] - SQL Anywhere
	[7] - BDB
	==============================================================================
	Enter choice (1): 3
	Hostname (localhost): localhost
	Port (3306): 3306
	Database name (ambari): ambari
	Username (ambari): ambari
	Enter Database Password (bigdata): 
	Re-enter password: 
	Configuring ambari database...
	Should ambari use existing default jdbc /usr/share/java/mysql-connector-java.jar [y/n] (y)? y
	Configuring remote database connection properties...
	WARNING: Before starting Ambari Server, you must run the following DDL directly from the database shell to create the schema: /var/lib/ambari-server/resources/Ambari-DDL-MySQL-CREATE.sql
	Proceed with configuring remote database connection properties [y/n] (y)? y
	Extracting system views...
	ambari-admin-2.7.3.0.139.jar
	....
	Ambari repo file contains latest json url http://public-repo-1.hortonworks.com/HDP/hdp_urlinfo.json, updating stacks repoinfos with it...
	Adjusting ambari-server permissions and ownership...
	Ambari Server 'setup' completed successfully.

#### 初始化数据库

	mysql -uambari -proot
	use ambari
	source /var/lib/ambari-server/resources/Ambari-DDL-MySQL-CREATE.sql

#### 启动Ambari Server

	ambari-server start

显示：

	[root@hdp21 yum.repos.d]# ambari-server start
	Using python  /usr/bin/python
	Starting ambari-server
	Ambari Server running with administrator privileges.
	Organizing resource files at /var/lib/ambari-server/resources...
	Ambari database consistency check started...
	Server PID at: /var/run/ambari-server/ambari-server.pid
	Server out at: /var/log/ambari-server/ambari-server.out
	Server log at: /var/log/ambari-server/ambari-server.log
	Waiting for server start...................
	Server started listening on 8080
	
	DB configs consistency check: no errors and warnings were found.
	Ambari Server 'start' completed successfully.

访问web console：

	192.168.3.21:8080

默认账号密码：admin / admin


![](Images/1.png)



## 对92.168.3.22进行操作

### 设置主机名

	sudo hostnamectl set-hostname hdp22

## 对92.168.3.23进行操作

### 设置主机名

	sudo hostnamectl set-hostname hdp23
	
	
## 部署HDP集群

进入hpd21服务器

访问web console：

	192.168.3.21:8080

默认账号密码：admin / admin

![](Images/2.png)

选择redhat7,填入以下内容：

	http://192.168.3.21/ambari/HDP/centos7/3.1.0.0-78
	http://192.168.3.21/ambari/HDP-GPL/centos7/3.1.0.0-78
	http://192.168.3.21/ambari/HDP-UTILS/centos7/1.1.0.22
	
	
![](Images/3.png)	
	
查看私钥

	[root@hdp21 /]# more ~/.ssh/id_rsa
	-----BEGIN RSA PRIVATE KEY-----
	MIIEpAIBAAKCAQEArfe3KGyruKVRh+c4rmhtc8v6VpJCyqEJyEOiyorvWsKFPhk1
	5QGS8tkRAOZSCc7Tq0SGWigKbOf3zaAPIlY3eZFuQlHBL8BJ9U1iEy0dme08+ICw
	rugYJt0WGd+SHy9YmKWTMBUlDDlPJGWhvZ6VPnmlcryX0DDEi1z++2oaPgYAoEXp
	mPyiahqmMXMTv5+SBvL5HKp387N9Gd6OHy2enX/SO4JzSGmsvmtWGCHZQ15A7vJN
	4LN6nRXDPHhM5CacmFl2f2Ll2KRHD1btPfRY21coEkHdrr4xqw1OMSueFeyFlFoG
	OzGeWh4fNexbETDYPfVkusIVSiXjmbMsorKk4wIDAQABAoIBACXB7e7IJz0cJGXq
	uVKhN6A6VUsU04pHCWh9xYdQQELilSi1MA1UGiC7HAWl94hJa4dwh/hpWFnqvs3S
	P3akX8Xy2otqUxOaGHM1KGQhFpuh8em9hJtxpExLD9DtxTkffYsj5J46+J9+kL7F
	kCUnlhzJKi+M3yZAb2gP9wd6L9fl+bsEnZAUWfEyYZvwHtep+B/9/YwwYfyVwjoK
	FEJ6fkCG9ekS2td7lgRGEvRXoAbUcc/Nwao+IIyZbQc5UkgPjqwB5aRyErSLug+7
	oGWRoIo9EenvMz9yumunqXa+plXY3/BdKgjM+RpmJ+DXC5vaOBcENQjfNuv7Xe+l
	C0n13gECgYEA5AzzpZj9Fvb8J2Eud4nRAZqWsc0akhpHDarflAvxHsAUU4uXgd9x
	EWnjJqGscLfpTGMUbL2T5UlYrSDSw+g5btpKI5OPbugaYcQJxITDVbT+mnU5CF5m
	kihedzKoQN/HRPWExttw/N9/O+obnQtwDBdX4TWqo6go5K+z8thijRMCgYEAw0nr
	mJOwdlQVzlJliDNYNvmkBp+nQXn31MzwUkELiE41PeUIiGe0HcfJtgrbjpV0gwMX
	ciCJEmX9IQn3bwreFVhl+KyfgJx7PA2TIcpu0Irja8uqJND0Ysd4CNc0Of+8Cpra
	iB8fVjANIxknMgUSgh8x4uN3oKhIQ4oNI6KNkvECgYEA2mlpJM2sf8UW5ORXa5dr
	vHyBHVLRwuI18OoZbaIuUYQnM4i7F2CX2tSt//7Rt7b8UvOFyUucDPo9WrrYAVc1
	Qm0AmqgltFPJiJ8Ht9Zrend5bRceA1zeIddV+hVjPZbVM79tZkMfKKEpK4ng3nzo
	2Du3J/ONNWbFm5iH4OrLIE8CgYBDjju40UZW+Kl+cRR8dD6M7fjm7a2/QcEQ1UBB
	RA0IPv2+Bo/YF/+/2ZAWsFTZUyw18gjXSxf/zFYvarI0sI641ZaY2qZmqDsdQBjo
	tI13CMS5Un/uMnaIATat4DxJ4ASdqon0UsfVWdQBkz7y17xaVQ49zbBhXDV7X7NB
	44aUEQKBgQC2PHrRzTk558v3K6Pgc9Eqc20g16jzgbR7IylAZL6C9KVjc2+nPLnm
	kRnEBty3DI/MRuCV6TsXPFgzBTx1JdJ0CNgOpxKFNJ48kPM0JGZuqgY17BLOqT+Z
	NS0Xs1l7FrfOaWUxRYx4voUBFCoA1pkPn8q5K9QsCcrIn/gTc7tpAQ==
	-----END RSA PRIVATE KEY-----
	
	
成功后显示：

![](Images/4.png)	

选择所需的组件

![](Images/5.png)	

选择部署的机器

![](Images/6.png)	

### 如果第9步出错，并生成了hdp.repo

ambari-hdp-1.repo：

	[HDP-3.1-repo-1]
	name=HDP-3.1-repo-1
	baseurl=
	
	
	path=/
	enabled=1
	gpgcheck=0
	[HDP-UTILS-1.1.0.22-repo-1]
	name=HDP-UTILS-1.1.0.22-repo-1
	baseurl=
	
	
	path=/
	enabled=1
	gpgcheck=0

处理步骤：

	Incase you have already at Step 9 and cannot proceed with ambari-server reset (as it invovles lots of Configs being added again , the below steps are for you )
	
	Preqrequesties : The cluster now is in Deployment step(step 9 ) and you have only retry button to press
	
	steps
	
	1) Stop ambari-server
	ambari-server stop
	
	2) login to Database
	mysql -uroot -proot
	
	3) use the below command to list out all the contents in repo_definition table :
	
	use ambari;
	 select * from repo_definition; 
	4) you can see the base_url will be empty for the all the Rows in the table
	
	5) Correct the base_url for every rows and update it using the command :
	
	update repo_definition set base_url='<YOUR BASE URL>' where id=<THE CORESPONDING ID>;
	for ex :
	
	update repo_definition set base_url='http://asnaik.example.com/localrepo/HDP-3.1' where id=9;
	6) after correcting all the base_url columns in repo_definition table and also delete the empty repos created by ambari from location /etc/yum.repos.d
	
	7) start ambari, 
	
	重启后在repo_definition表中会新增 HDP-GPL ，更改base_url
	
	8）Login to UI and press retry button, The Installation will work as smooth as it can be.


数据库操作详情：

	MariaDB [ambari]> select * from repo_definition;
	+----+------------+-----------+--------------------+----------+--------------+------------+-------------+---------+
	| id | repo_os_id | repo_name | repo_id            | base_url | distribution | components | unique_repo | mirrors |
	+----+------------+-----------+--------------------+----------+--------------+------------+-------------+---------+
	| 76 |         59 | HDP       | HDP-3.1            |          | NULL         | NULL       |           0 | NULL    |
	| 77 |         59 | HDP-UTILS | HDP-UTILS-1.1.0.22 |          | NULL         | NULL       |           0 | NULL    |
	+----+------------+-----------+--------------------+----------+--------------+------------+-------------+---------+
	2 rows in set (0.00 sec)
	
	MariaDB [ambari]> update repo_definition set base_url='http://192.168.3.21/ambari/HDP/centos7/3.1.0.0-78' where id=76;
	Query OK, 1 row affected (0.00 sec)
	Rows matched: 1  Changed: 1  Warnings: 0
	
	MariaDB [ambari]> select * from repo_definition;
	+----+------------+-----------+--------------------+---------------------------------------------------+--------------+------------+-------------+---------+
	| id | repo_os_id | repo_name | repo_id            | base_url                                          | distribution | components | unique_repo | mirrors |
	+----+------------+-----------+--------------------+---------------------------------------------------+--------------+------------+-------------+---------+
	| 76 |         59 | HDP       | HDP-3.1            | http://192.168.3.21/ambari/HDP/centos7/3.1.0.0-78 | NULL         | NULL       |           0 | NULL    |
	| 77 |         59 | HDP-UTILS | HDP-UTILS-1.1.0.22 |                                                   | NULL         | NULL       |           0 | NULL    |
	+----+------------+-----------+--------------------+---------------------------------------------------+--------------+------------+-------------+---------+
	2 rows in set (0.00 sec)
	
	MariaDB [ambari]> update repo_definition set base_url='http://192.168.3.21/ambari/HDP-UTILS/centos7/1.1.0.22' where id=77;
	Query OK, 1 row affected (0.01 sec)
	Rows matched: 1  Changed: 1  Warnings: 0
	
	MariaDB [ambari]> 
	MariaDB [ambari]> 
	MariaDB [ambari]> select * from repo_definition;
	+----+------------+-----------+--------------------+-------------------------------------------------------+--------------+------------+-------------+---------+
	| id | repo_os_id | repo_name | repo_id            | base_url                                              | distribution | components | unique_repo | mirrors |
	+----+------------+-----------+--------------------+-------------------------------------------------------+--------------+------------+-------------+---------+
	| 76 |         59 | HDP       | HDP-3.1            | http://192.168.3.21/ambari/HDP/centos7/3.1.0.0-78     | NULL         | NULL       |           0 | NULL    |
	| 77 |         59 | HDP-UTILS | HDP-UTILS-1.1.0.22 | http://192.168.3.21/ambari/HDP-UTILS/centos7/1.1.0.22 | NULL         | NULL       |           0 | NULL    |
	+----+------------+-----------+--------------------+-------------------------------------------------------+--------------+------------+-------------+---------+
	2 rows in set (0.00 sec)
	
	MariaDB [ambari]> update repo_definition set base_url='http://192.168.3.21/ambari/HDP-GPL/centos7/3.1.0.0-78' where id=151;
	Query OK, 1 row affected (0.00 sec)
	Rows matched: 1  Changed: 1  Warnings: 0
	
	MariaDB [ambari]> 
	MariaDB [ambari]> select * from repo_definition;
	+-----+------------+-----------+--------------------+-------------------------------------------------------+--------------+------------+-------------+---------+
	| id  | repo_os_id | repo_name | repo_id            | base_url                                              | distribution | components | unique_repo | mirrors |
	+-----+------------+-----------+--------------------+-------------------------------------------------------+--------------+------------+-------------+---------+
	| 126 |        109 | HDP       | HDP-3.1            | http://192.168.3.21/ambari/HDP/centos7/3.1.0.0-78     | NULL         | NULL       |           0 | NULL    |
	| 127 |        109 | HDP-UTILS | HDP-UTILS-1.1.0.22 | http://192.168.3.21/ambari/HDP-UTILS/centos7/1.1.0.22 | NULL         | NULL       |           0 | NULL    |
	| 151 |        109 | HDP-GPL   | HDP-3.1-GPL        | http://192.168.3.21/ambari/HDP-GPL/centos7/3.1.0.0-78 | NULL         | NULL       |           0 | NULL    |
	+-----+------------+-----------+--------------------+-------------------------------------------------------+--------------+------------+-------------+---------+
	3 rows in set (0.00 sec)
	
	
执行成功：

![](Images/7.png)	

查看jps：

	[root@hdp21 ~]# jps
	2276 AMSApplicationServer
	2214 HMaster
	1544 SupportToolServer
	9065 HRegionServer
	11402 TimelineReaderServer
	12426 NodeManager
	1515 AmbariServer
	7884 JobHistoryServer
	11565 ActivityAnalyzerFacade
	3633 
	3922 DataNode
	13299 Jps
	12054 ZeppelinServer
	7479 ApplicationHistoryServer
	3735 QuorumPeerMain
	8727 HMaster
	8123 ResourceManager
	4191 NameNode
	8415 SecondaryNameNode


管理界面：

![](Images/8.png)


### 测试

切换到hdfs用户才能执行

	[root@hdp21 ~]# su hdfs
	[hdfs@hdp21 root]$ hdfs dfs -mkdir /input
	
	#造数据
	[hdfs@hdp21 /]$ sudo chmod -R 777 /hdfs_user/
	[hdfs@hdp21 /]$ cd /hdfs_user/
	[hdfs@hdp21 root]$ hdfs dfs -ls /
	[hdfs@hdp21 hdfs_user]$ vi data.txt
	[hdfs@hdp21 hdfs_user]$ hdfs dfs -put data.txt /input
	[hdfs@hdp21 hdfs_user]$ hdfs dfs -cat /input/data.txt
	
执行wordcount demo	

	[hdfs@hdp21 hdfs_user]$ cd /usr/hdp/3.1.0.0-78/hadoop-mapreduce/hadoop-mapreduce-examples
	[hdfs@hdp21 hadoop-mapreduce]$ hadoop jar hadoop-mapreduce-examples.jar wordcount /input/data.txt /output/wc
	

## 节点扩展

以扩展datanode为例：

在hdp22和hdp23上创建目录

	mkdir -p /var/lib/ambari-agent/data

![](Images/9.png)

步骤跟前面一样。


![](Images/10.png)
选择datanode



### 安装错误

	关于ambri hst agent注册失败错误，查看log:
	=============
	INFO 2017-09-21 10:52:33,435 security.py:178 - Server certificate not exists, downloading
	INFO 2017-09-21 10:52:33,435 security.py:191 - Downloading server cert from https://ambari-test1.com:9440/cert/ca/
	ERROR 2017-09-21 10:52:33,510 ServerAPI.py:137 - POST https://ambari-test1.com:9441/api/v1/register failed. (SSLError(1, u'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:579)'),)
	============
	这个错误是因为python-2.7.5-e58版本默认是使用ssl验证，解决这个问题，修改/etc/python/cert-verification.cfg：
	[https]
	verify=disable
	

或者调大内存

### 成功界面
	
![](Images/11.png)