# Docker安装mysql


启动启动 

    docker run --name mysql -p3306:3306 -e MYSQL_ROOT_PASSWORD=root -d mysql

进入容器

	docker exec -it mysql bash


设置远程的授权等信息:


	mysql -uroot -p
	 
	 grant all privileges on *.* to root@"%" identified by "root" with grant option; 
	 
	 ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY 'root';
	  
	 flush privileges;