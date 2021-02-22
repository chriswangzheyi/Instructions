# Docker安装Keyloak

参考：https://blog.csdn.net/letterss/article/details/108832557


## 创建网络


	docker network create -d bridge cloud


## 安装mysql

###启动启动 

    docker run --name mysql -p3306:3306 -e MYSQL_ROOT_PASSWORD=root -d mysql

### 进入容器

	docker exec -it mysql bash


### 设置远程的授权等信息:


	mysql -uroot -p
	 
	 grant all privileges on *.* to root@"%" identified by "root" with grant option; 
	 
	 ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY 'root';
	  
	 flush privileges;


### 创建keycloak数据库

	CREATE DATABASE IF NOT EXISTS keycloak DEFAULT CHARSET utf8 COLLATE utf8_general_ci;


## 安装keyloak

	docker run --name keycloak    --restart=always     --network cloud     -p 8010:8080   -e KEYCLOAK_USER=admin    -e KEYCLOAK_PASSWORD=admin    -e DB_VENDOR=mysql   -e DB_ADDR=47.112.142.231     -e DB_PORT=3306    -e DB_DATABASE=keycloak    -e DB_USER=root     -e DB_PASSWORD=1qa2ws#ED     -e JDBC_PARAMS='connectTimeout=90&useSSL=false'     -d jboss/keycloak


### 修改配置

因为模式是使用的https 所以需要进行登录系统修改

登录数据库进行修改

	mysql -uroot -p

	use keycloak
	update REALM set ssl_required='NONE' where id = 'master';


## 访问keyloak

### 重启容器

	docker restart  keycloak


### 访问

	http://47.112.142.231:8010/


账号密码都是admin