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


解释：

a1.sources.r1.type 指定了类型

a1.sources.r1.bind = 0.0.0.0 4个0表示通配

a1.channels.c1.capacity = 100  通道里面最多容纳100个事件

a1.sinks.k1.channel=c1 只能指定一个channel

## 启动flume

	/root/flumeDemo/flume/bin/flume-ng agent -n a1 -f /root/flumeDemo/flume/conf/demo.conf -c /root/flumeDemo/flume/conf  -Dflume.root.logger=INFO,console

解释：

-c 指定flume自身的配置文件所在目录

-f 指定conf文件路径

-n 指定agent的名字


## 验证

新开一个窗口，执行命令：

	telnet localhost 8888


