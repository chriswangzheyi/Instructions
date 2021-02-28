#06 安装Flink

参考：https://zhuanlan.zhihu.com/p/131588371

## 解压

	tar -zxvf flink-1.12.1-bin-scala_2.12.tgz
	mv flink-1.12.1 /usr/local/flink

## 修改端口号

	cd /usr/local/flink/conf
	vim masters

修改端口为8381

	cd /usr/local/flink/conf
	vim flink-conf.yaml

设置

	rest.port: 8381



## 启动

	cd /usr/local/flink/bin
	./start-cluster.sh 

对于单节点设置，Flink是开箱即用的，即不需要更改默认配置，直接启动即可。


## 验证

JPS查看

	2672 StandaloneSessionClusterEntrypoint
	3096 TaskManagerRunner


网页

	http://192.168.195.150:8381/
