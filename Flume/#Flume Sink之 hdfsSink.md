#Flume Sink之 hdfsSink

## 概念

在本地文件系统中存储事件。每隔指定时长生成文件保存这段时间内收集到的日志信息到HDFS中。

##  创建config

vi /root/flumeDemo/flume/conf/s_hdfs.conf

	a1.sources = r1
	a1.channels = c1
	a1.sinks = k1
	
	#define source
	a1.sources.r1.type = seq
	a1.sources.r1.totalEvents = 100000000
	
	# define channel
	a1.channels.c1.type = memory
	a1.channels.c1.capacity = 100
	
	# define sink
	a1.sinks.k1.type = hdfs
	#控制hdfs上的目录
	a1.sinks.k1.hdfs.path = hdfs://47.112.142.231:9000/test/lwq/%y-%m-%d
	#文件前缀
	a1.sinks.k1.hdfs.filePrefix = events-
	#round控制的是目录的滚动
	a1.sinks.k1.hdfs.round = true
	a1.sinks.k1.hdfs.roundValue = 5
	a1.sinks.k1.hdfs.roundUnit = second
	
	#文件滚动
	a1.sinks.k1.hdfs.rollInterval = 1
	a1.sinks.k1.hdfs.rollSize = 1024
	a1.sinks.k1.hdfs.rollCount = 100
	
	#设置使用本地时间戳，否则需要单独在event header设置时间戳。
	a1.sinks.k1.hdfs.useLocalTimeStamp = true
	
	#允许hdfs操纵的时间，比如open、flush、write、close，
	#如果在是定时间段内还没有完成，抛异常。
	a1.sinks.k1.hdfs.callTimeout=10000
	
	#控制文件类型SequenceFile, DataStream-文本文件 CompressedStream-压缩流文件
	a1.sinks.k1.hdfs.fileType = DataStream 
	
	#bind
	a1.sources.r1.channels=c1
	a1.sinks.k1.channel=c1

## 启动

	/root/flumeDemo/flume/bin/flume-ng agent -f /root/flumeDemo/flume/conf/s_hdfs.conf  -c /root/flumeDemo/flume/conf -n a1


