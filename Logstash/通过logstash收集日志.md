# 通过logstash收集日志


参考：https://www.cnblogs.com/yanjieli/p/11187573.html

## 步骤


### 创建测试目录

使用非root账号：

	mkdir -p /es/data/{messages,secure}
       

### 创建配置文件

	vi /es/logstash-7.7.0/config/logstash-test.conf

插入

	input {
	    file {
	        path => "/es/data/messages/1.data"
	        type => "systemlog"
	        start_position => "beginning/1.data"
	        stat_interval => "3"
	    }
	    file {
	        path => "/es/data/secure"
	        type => "securelog"
	        start_position => "beginning"
	        stat_interval => "3"
	    }
	}
	
	output {
	    if [type] == "systemlog" { 
	        elasticsearch {
	            hosts => ["192.168.195.150:9200"]
	            index => "system-log-%{+YYYY.MM.dd}"
	        }
	    }
	    if [type] == "securelog" { 
	        elasticsearch {
	            hosts => ["192.168.195.150:9200"]
	            index => "secure-log-%{+YYYY.MM.dd}"
	        }
	    }
	}


### 启动logstash

	cd /es/logstash-7.7.0/bin
	./logstash -f /es/logstash-7.7.0/config/logstash-test.conf

### 写入测试数据

	echo "test" >> /es/data/messages/1.data 
	echo "test" >> /es/data/secure/1.data 





## 查看kibana

### 设置索引

![](../Images/2.png)

![](../Images/3.png)

![](../Images/4.png)

### 统计数据

![](../Images/5.png)