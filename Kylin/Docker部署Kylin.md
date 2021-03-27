# Docker部署Kylin

## 拉取镜像

	docker pull apachekylin/apache-kylin-standalone:3.0.1


## 运行docker


	docker run -d \
	-m 8G \
	-p 9009:9009 \
	-p 10033:10033 \
	-p 39283:39283 \
	-p 46709:46709 \
	-p 50070:50070 \
	-p 38999:38999 \
	-p 8088:8088 \
	-p 13562:13562 \
	-p 50010:50010 \
	-p 50075:50075 \
	-p 7070:7070 \
	-p 8030:8030 \
	-p 8031:8031 \
	-p 8032:8032 \
	-p 8033:8033 \
	-p 10020:10020 \
	-p 9092:9092 \
	-p 50020:50020 \
	-p 40997:40997 \
	-p 2181:2181 \
	-p 8998:8998 \
	-p 33319:33319 \
	-p 8040:8040 \
	-p 9000:9000 \
	-p 7337:7337 \
	-p 16010:16010 \
	-p 8042:8042 \
	-p 3306:3306 \
	-p 9005:9005 \
	-p 33423:33423 \
	apachekylin/apache-kylin-standalone:3.0.1


## 界面

	livy web http://47.112.142.231:8998/
	Kylin 页面：http://47.112.142.231:7070/kylin/login – 默认账号：ADMIN 默认密码:KYLIN
	HDFS NameNode 页面：http://47.112.142.231:50070/dfshealth.html#tab-overview
	YARN ResourceManager 页面：http://47.112.142.231:8088/cluster
	HBase 页面：http://47.112.142.231:16010/master-status
	
	
kylin的登陆页面可以在log中看到：

	Web UI is at http://6a96f4815e57:7070/kylin
	
如果看不到登陆页面，需要在bin文件下，./kylin.sh start来启动
