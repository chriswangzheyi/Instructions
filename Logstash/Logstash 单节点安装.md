# Logstash 单节点安装

## 安装包下载

官网下载较慢，可以通过华为镜像

	https://mirrors.huaweicloud.com/logstash/7.7.0/


## 安装步骤

	tar -zxvf logstash-7.7.0.tar.gz


## 测试

	cd /root/logstash-7.7.0/bin
	

	./logstash -e 'input { stdin { } } output { stdout {} }'


-e可以直接从命令行指定配置。通过在命令行指定配置，可以快速测试配置，而无需在迭代之间编辑文件。示例中的管道从标准输入stdin获取输入数据，并以结构化格式将输入数据移动到标准输出stdout 。

运行之后等待一会儿之后可以看到如下日志，出现Successfully started Logstash API endpoint {:port=>9600}标志着启动成功


管道启动完之后就可以在控制台发送信息了，控制台会返回我们输入的信息，如下：

输入：hello world

![](../Images/1.png)