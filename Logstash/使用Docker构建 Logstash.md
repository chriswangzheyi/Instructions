# 使用Docker构建 Logstash


## 搭建Logstash

###拉取镜像
docker pull logstash

###启动logstash
docker run -it -d -p 4567:4567 --name logstash logstash


###创建配置文件

	mkdir /root/logstash_test
	vi  /root/logstash_test/logstash.conf

	#插入
	input{
	        tcp {
	                mode => "server"
	                host => "0.0.0.0"
	                port => 4567
	                codec => json_lines //需要安装logstash-codec-json_lines插件
	        }
	}
	output{
	        elasticsearch{
	                hosts=>["47.112.142.231:9200"]
	                index => "springboot"
	                }
	        stdout{codec => rubydebug}
	}
	

如果Logstash没有安装logstash-codec-json_lines插件，通过以下命令安装：	
 
	cd /usr/share/logstash/bin
	logstash-plugin install logstash-codec-json_lines


### 将文件拷贝到docker 容器中

	 docker cp /root/logstash_test/logstash.conf logstash:/etc/logstash


### 进入容器
	
	docker exec -it logstash /bin/sh

### 删除不必要文件

	rm /var/lib/logstash/.lock