# logstash写入到kafka和从kafka读取日志


## logstash从kafka读取日志到ES

### 配置文件

vi kafka-logstash.conf

插入

	input {
	  kafka {
	    codec => "json"
	    topics => ["test"]
	    bootstrap_servers => [" 192.168.195.150:9092"]
	    group_id => "logstash-g1"
	  }
	}
	output {
	  elasticsearch {
	    hosts => ["192.168.195.150:9200"]
	    index => "kafka-%{+YYYY.MM.dd}"
	  }
	}


### 测试


	cd /es/logstash-7.7.0/bin
	./logstash -f /es/logstash-7.7.0/config/kafka-logstash.conf



### 启动kafka

	./kafka-console-producer.sh --broker-list localhost:9092 --topic test

输入测试样本：

	[{"国家":"中国","人口":13.83,"国土面积":963},{"国家":"美国","人口":3.27,"国土面积":962}]



