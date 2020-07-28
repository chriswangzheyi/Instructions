# Docker 安装 Hive

参考 https://github.com/big-data-europe/docker-hive


## 下载相关安装包

	https://github.com/big-data-europe/docker-hive

## 启动

	docker-compose up -d

## 验证

	docker-compose exec hive-server bash

	/opt/hive/bin/beeline -u jdbc:hive2://47.112.142.231:10000

进入hive后

	CREATE TABLE pokes (foo INT, bar STRING);
	LOAD DATA LOCAL INPATH '/opt/hive/examples/files/kv1.txt' OVERWRITE INTO TABLE pokes;