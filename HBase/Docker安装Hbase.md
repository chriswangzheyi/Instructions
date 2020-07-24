# Docker 安装Hbase


## 拉取镜像

	docker pull harisekhon/hbase


## 启动
    docker run -d -p 2181:2181 -p 8080:8080 -p 8085:8085 -p 9090:9090 -p 9095:9095 -p 16000:16000 -p 16010:16010 -p 16201:16201 -p 16301:16301 --name hbase harisekhon/hbase

## 访问管理页面

	http://47.112.142.231:16010/master-status