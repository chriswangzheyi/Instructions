# Flume Exe 实时搜集

---

## 创建config文件

vi /root/flumeDemo/flume/conf/r_exe.conf

	a1.sources = r1
	a1.channels = c1
	a1.sinks = k1
	
	#define source
	a1.sources.r1.type = exec
	a1.sources.r1.command = tail -F /root/testlogs/1.log
	
	# define channel
	a1.channels.c1.type = memory
	a1.channels.c1.capacity = 100
	
	# define sink
	a1.sinks.k1.type = logger
	
	#bind
	a1.sources.r1.channels=c1
	a1.sinks.k1.channel=c1



## 启动

	/root/flumeDemo/flume/bin/flume-ng agent -n a1 -f /root/flumeDemo/flume/conf/r_exe.conf -c /root/flumeDemo/flume/conf -Dflume.root.logger=INFO,console


## 测试

	for ((x=1;x<100;x=$x+1)) ;do sleep 1 ; echo $x ; echo tom$x >> /root/testlogs/1.log ; done