#Flume Seq-Stress-Spoold源 Demo

## 序列源

### 创建conf

vi /root/flumeDemo/flume/conf/r_seq.conf

	a1.sources = r1
	a1.channels = c1
	a1.sinks = k1
	
	#define source
	a1.sources.r1.type = seq
	#事件总数。
	a1.sources.r1.totalEvents = 1000
	
	# define channel
	a1.channels.c1.type = memory
	a1.channels.c1.capacity = 100
	
	# define sink
	a1.sinks.k1.type = logger
	
	#bind
	a1.sources.r1.channels=c1
	a1.sinks.k1.channel=c1

### 启动

	/root/flumeDemo/flume/bin/flume-ng agent -n a1 -f /root/flumeDemo/flume/conf/r_seq.conf -c /root/flumeDemo/flume/conf -Dflume.root.logger=INFO,console


![](../Images/3.png)


## 压力源

适用于做压力测试

vi /root/flumeDemo/flume/conf/r_stress.conf

	a1.sources = r1
	a1.channels = c1
	a1.sinks = k1
	
	#define source
	a1.sources.r1.type = org.apache.flume.source.StressSource
	a1.sources.r1.size = 10240
	a1.sources.r1.maxTotalEvents = 1000
	
	# define channel
	a1.channels.c1.type = memory
	a1.channels.c1.capacity = 100
	
	# define sink
	a1.sinks.k1.type = logger
	
	#bind
	a1.sources.r1.channels=c1
	a1.sinks.k1.channel=c1

### 启动

	/root/flumeDemo/flume/bin/flume-ng agent -n a1 -f /root/flumeDemo/flume/conf/r_stress.conf -c /root/flumeDemo/flume/conf -Dflume.root.logger=INFO,console


## 监控目录的源  spooldir

针对指定的目录进行监控，一旦新文件进入，就进行数据提取，数据提取完成的文件重命名到.completed文件。

因此，要求文件必须在目录外创建，通过mv或拷贝的方式放置到指定目录中。若对目录内的文件进行append工作，无法收集到新的日志。


vi /root/flumeDemo/flume/conf/r_spooldir.conf

	a1.sources = r1
	a1.channels = c1
	a1.sinks = k1
	
	#define source
	a1.sources.r1.type = spooldir
	a1.sources.r1.spoolDir = /root/testlogs
	
	# define channel
	a1.channels.c1.type = memory
	a1.channels.c1.capacity = 100
	
	# define sink
	a1.sinks.k1.type = logger
	
	#bind
	a1.sources.r1.channels=c1
	a1.sinks.k1.channel=c1

### 启动

	/root/flumeDemo/flume/bin/flume-ng agent -n a1 -f /root/flumeDemo/flume/conf/r_spooldir.conf -c /root/flumeDemo/flume/conf -Dflume.root.logger=INFO,console

启动后就会监控 /root/testlogs下的所有文件的改变

监控结果是不能append的。

