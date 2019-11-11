# Docker 搭建Logstash(单机测试)

## 搭建步骤

###拉取镜像

	docker pull logstash

###启动logstash

	docker run -it -d --name logstash logstash

###创建配置文件

	mkdir /root/logstash_test
	vi  /root/logstash_test/logstash.conf

	#插入
	input { stdin {} }
	output {
	  elasticsearch {
	    action => "index"
	    hosts => "http://47.112.142.231:9200"
	    index => "my_log"#在es中的索引
	  }
	  # 这里输出调试，正式运行时可以注释掉
	  stdout {
	      codec => json_lines
	  }
	}


### 将文件拷贝到docker 容器中

	 docker cp /root/logstash_test/logstash.conf logstash:/etc/logstash

###启动logstash

	# 进入容器
	docker exec -it logstash /bin/sh
	
	
	cd /etc/logstash

	logstash -f logstash.conf

注意：如果看到这样的报错信息 Logstash could not be started because there is already another instance using the configured data directory.  If you wish to run multiple instances, you must change the "path.data" setting.删除/var/lib/logstash 中的 .lock文件即可

	rm /var/lib/logstash/.lock
	

然后再次执行就可以了。


## 测试

	启动好Logstash、kibana和ElasticSearh

在kibana的Devtool中，执行下面命令即可：

	GET /my_log/_search 




	


