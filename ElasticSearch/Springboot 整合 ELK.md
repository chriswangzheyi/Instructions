# Springboot整合ELK

# 准备容器

### 构建dokcer-compose 

mkdir -p /root/elk

vi /root/elk/docker-compose.yml

	version: '2'
	services:
	  elasticsearch:
	    image: elasticsearch
	    environment:
	      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
	    volumes:
	      - $PWD/elasticsearch/data:/usr/share/elasticsearch/data
	    hostname: elasticsearch
	    restart: always
	    ports:
	      - "9200:9200"
	      - "9300:9300"
	  kibana:
	    image: kibana
	    environment:
	      - ELASTICSEARCH_URL=http://elasticsearch:9200 #elasticsearch查询接口地址
	    hostname: kibana
	    depends_on:
	      - elasticsearch  #后于elasticsearch启动
	    restart: always
	    ports:
	      - "5601:5601"
	  logstash:
	    image: logstash
	    command: logstash -f /etc/logstash/conf.d/logstash.conf  #logstash 启动时使用的配置文件
	    volumes:
	      - $PWD/logstash/conf.d:/etc/logstash/conf.d  #logstash 配文件位置
	      - $PWD/logst:/tmp
	    hostname: logstash
	    restart: always
	    depends_on:
	      - elasticsearch  #后于elasticsearch启动
	    ports:
	      - "7001-7005:7001-7005"
	      - "4560:4560"
	      - "9600:9600"





### 配置文件
mkdir -p /root/elk/logstash/conf.d

vi /root/elk/logstash/conf.d/logstash.conf

	input {
	    tcp {
	        mode => "server"
	        host => "0.0.0.0"
	        port => 4560
	        codec => json_lines
	    }
	}
	output{
	  elasticsearch {
	    hosts => ["elasticsearch:9200"]    
	    action => "index"
	    index => "applog"
	    }
	  stdout {
	    codec => rubydebug
	    }
	}



### 操作容器

	## 启动容器
	docker-compose up -d 

	## 查看状态
	docker ps -a	

	## 停止容器组
	docker-compose down


# Springboot

## 项目结构

![](Images/13.png)


**HelloController:**

	package com.wzy.controller;
	
	import org.slf4j.Logger;
	import org.slf4j.LoggerFactory;
	import org.springframework.web.bind.annotation.GetMapping;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	public class HelloController {
	
	    private final static Logger logger = LoggerFactory.getLogger( HelloController.class );
	
	    @GetMapping("/")
	    public String hi() {
	        logger.info( "it is calling"   );
	        return "hi ";
	    }
	}


**SpringbootElkApplication:**

	package com.wzy;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootElkApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootElkApplication.class, args);
	    }
	
	}


**application.yml:**

	server:
	  port: 8081


**logback-spring.xml:**

	<?xml version="1.0" encoding="UTF-8"?>
	<configuration>
	    <include resource="org/springframework/boot/logging/logback/base.xml" />
	
	    <appender name="LOGSTASH" class="net.logstash.logback.appender.LogstashTcpSocketAppender">
	        <!-- logstash 输入地址  与logstash.conf 配置文件的input对应-->
	        <destination>47.112.142.231:4560</destination>
	        <encoder charset="UTF-8" class="net.logstash.logback.encoder.LogstashEncoder" />
	    </appender>
	
	    <root level="INFO">
	        <appender-ref ref="LOGSTASH" />
	        <appender-ref ref="CONSOLE" />
	    </root>
	</configuration>

**pom.xml:**

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.2.0.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>springboot_elk</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_elk</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-web</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-test</artifactId>
	            <scope>test</scope>
	        </dependency>
	
	        <!-- logback -->
	        <dependency>
	            <groupId>ch.qos.logback</groupId>
	            <artifactId>logback-classic</artifactId>
	        </dependency>
	
	        <dependency>
	            <groupId>net.logstash.logback</groupId>
	            <artifactId>logstash-logback-encoder</artifactId>
	            <version>5.2</version>
	        </dependency>
	    </dependencies>
	
	    <build>
	        <plugins>
	            <plugin>
	                <groupId>org.springframework.boot</groupId>
	                <artifactId>spring-boot-maven-plugin</artifactId>
	            </plugin>
	        </plugins>
	    </build>
	</project>


# 测试

启动springboot, 访问8081端口。

进入kibana

	GET /applog/_search 