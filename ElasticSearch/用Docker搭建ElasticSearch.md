**用Docker构建ElasticSearch**

lucene_version" : "6.6.1

**1.启动服务**

    docker run -e ES_JAVA_OPTS="-Xms256m -Xmx256m" -d -p 9200:9200 -p 9300:9300  -p 9100:9100  --name myes  docker.io/elasticsearch:6.6.1 


**2.验证**

访问{ip}:9200 

    http://47.112.142.231:9200/

得到:

![](Images/1.png)



## 报错

max virtual memory areas vm.max_map_count [65530] is too low, increase to at least [262144]

解决：

切换到root用户

执行命令：

	sysctl -w vm.max_map_count=262144


上述方法修改之后，如果重启虚拟机将失效，所以：

解决办法：

在   /etc/sysctl.conf文件最后添加一行

	vm.max_map_count=262144

即可永久修改