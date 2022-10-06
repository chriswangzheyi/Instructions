# MySQL 单节点安装


## 安装

### 配置源

	wget https://dev.mysql.com/get/mysql80-community-release-el7-3.noarch.rpm 
	yum localinstall  mysql80-community-release-el7-3.noarch.rpm
	sudo rpm --import https://repo.mysql.com/RPM-GPG-KEY-mysql-2022
	yum install mysql-community-client mysql-community-server


## 服务操作

###启动
service mysqld start 

###重启
service mysqld restart 

### 停掉
service mysqld stop 

### 开机启动
systemctl start mysqld.service

## 进入Mysql

### 查看密码

需要先启动mysql

查看临时密码：

	cat /var/log/mysqld.log | grep 'password'

A temporary password is generated for root@localhost: #Y!o2I>/7op*



### 登录mysql

	mysql -uroot -p


### 远程授权

#### 第一次登陆需要修改密码 （可选项）
ALTER USER USER() IDENTIFIED BY '1qa2ws#ED';


#### 操作步骤

	use mysql;
	
	select host from user where user='root';
	
	update user set host = '%' where user ='root';
	
	ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY '1qa2ws#ED';

	grant all privileges on *.* to 'root'@'%'; 
		  
	flush privileges;
	
### 关闭防火墙

	systemctl stop firewalld.service
	systemctl disable firewalld.service