# Docer 搭建InfluxDB


## 拉取镜像并启动

	docker run -d -p 8083:8083 -p 8086:8086 --name my_influxdb influxdb

## 配置远程访问

进入influxdb容器

	docker exec -it my_influxdb bash

打开influxdb控制台

	cd /usr/bin
	./influx

创建admin用户

	# 查看所有用户
	> show users
	user admin
	---- -----
	> 
	# 创建一个root用户，设置密码为newpwd，主要不要使用双引号" 括起来，不然会报错
	> create user "root" with password 'newpwd'
	> 
	# 再次查看用户信息，发现admin为false，说明还要设置一下权限。
	> show users
	user admin
	---- -----
	root false
	> 
	# 删除root用户
	> drop user root
	> 
	> show users
	user admin
	---- -----
	> 
	# 重新设置root用户，并设置带上所有权限
	> create user "root" with password 'newpwd' with all privileges
	> 
	# 发现admin权限为true了，那么admin的用户就创建好了。
	> show users
	user admin
	---- -----
	root true

## 容器内安装vi

	apt-get update
	apt-get install vim

## 在配置文件启用认证

在容器内执行：

	vim /etc/influxdb/influxdb.conf
	
插入：

	[HTTP]
		auth-enable =true

## 验证

配置完毕之后，重启influxdb服务即可。

	influx -username 'root' -password 'newpwd'