# Docker安装clickhouse 单机版

## server安装命令

	docker run -d \
	--name clickhouse-server \
	-p 9000:9000 \
	-p 8123:8123 \
	-p 9009:9009 \
	--ulimit nofile=262144:262144 \
	yandex/clickhouse-server


## 连接

###进入docker容器

	docker exec -it {container_id} bash


### 连接CK server

	cd	/bin
 	clickhouse-client  -m

	#显示
	ClickHouse client version 20.12.4.5 (official build).
	Connecting to localhost:9000 as user default.
	Connected to ClickHouse server version 20.12.4 revision 54442.



## 基本命令
	
	show databases;
	

## 远程访问

### clickhouse的配置文件拷贝出来

	docker cp clickhouse-server:/etc/clickhouse-server/ /etc/clickhouse-server/

### 修改命令

修改 /etc/clickhouse-server/config.xml 中 65行 注释去掉

	<listen_host>::</listen_host>

###
用自定义配置文件启动容器

	docker run -d --name docker-clickhouse --ulimit nofile=262144:262144 -p 8123:8123 -p 9000:9000 -p 9009:9009 -v /etc/clickhouse-server/config.xml:/etc/clickhouse-server/config.xml yandex/clickhouse-server
