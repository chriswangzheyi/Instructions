# Flume NetCat源Demo

## 修改config

创建文件：

	cp /root/flumeDemo/flume/conf/flume-conf.properties.template /root/flumeDemo/flume/conf/demo.conf
 
修改config：

	vi /root/flumeDemo/flume/conf/demo.conf

修改如下：

	a1.sources = r1
	a1.channels = c1
	a1.sinks = k1
	
	#define source
	a1.sources.r1.type = netcat
	a1.sources.r1.bind = 0.0.0.0
	a1.sources.r1.port = 8888
	
	# define channel
	a1.channels.c1.type = memory
	a1.channels.c1.capacity = 100
	
	# define sink
	a1.sinks.k1.type = logger
	
	#bind
	a1.sources.r1.channels=c1
	a1.sinks.k1.channel=c1

