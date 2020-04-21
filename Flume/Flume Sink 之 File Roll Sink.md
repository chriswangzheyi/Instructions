# Flume Sink 之 File Roll Sink

## 概念

在本地文件系统中存储事件。每隔指定时长生成文件保存这段时间内收集到的日志信息。

##  创建config

vi /root/flumeDemo/flume/conf/s_filerole.conf

	a1.sources = r1
	a1.channels = c1
	a1.sinks = k1
	
	#define source
	a1.sources.r1.type = seq
	a1.sources.r1.totalEvents = 1000
	
	# define channel
	a1.channels.c1.type = memory
	a1.channels.c1.capacity = 100
	
	# define sink
	a1.sinks.k1.type = file_roll
	#控制滚动间隔(秒)
	a1.sinks.k1.sink.rollInterval = 1
	
	#目录必须先创建
	a1.sinks.k1.sink.directory = /root/testlogs
	
	#bind
	a1.sources.r1.channels=c1
	a1.sinks.k1.channel=c1


## 启动

	/root/flumeDemo/flume/bin/flume-ng agent -f /root/flumeDemo/flume/conf/s_filerole.conf  -c /root/flumeDemo/flume/conf -n a1