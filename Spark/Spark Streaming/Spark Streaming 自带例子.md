# Spark Streaming 自带例子

## 窗口一
	#按照nc,非必须
	yum install -y nc
	
	#连接9999端口
	nc -l -p 9999

## 窗口二

	cd /home/hadoop_spark/spark-3.0.0-bin-hadoop3.2/bin
	 ./run-example streaming.NetworkWordCount 192.168.2.101 9999
	 
	 
## 测试

在窗口一当中输入文本，就可以在窗口二完成统计。注意，gpu核数需大于等于2.


### 窗口一

	[hadoop_spark@wangzheyi root]$ nc -l -p 9999
	hello world
	hello my test
	
### 窗口二

	-------------------------------------------
	Time: 1599472694000 ms
	-------------------------------------------
	(hello,1)
	(world,1)

	-------------------------------------------
	Time: 1599472702000 ms
	-------------------------------------------
	(hello,1)
	(my,1)
	(test,1)
	
可以看到，两次的结果没有进行累加。