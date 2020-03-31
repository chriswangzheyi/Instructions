# Springboot 整合 Kafka

## Producer

![](../Images/4.png)

### KafkaProducer

	package com.wzy.springboot_kafka_producer.producer;
	
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.kafka.core.KafkaTemplate;
	import org.springframework.scheduling.annotation.EnableScheduling;
	import org.springframework.scheduling.annotation.Scheduled;
	import org.springframework.stereotype.Component;
	import org.springframework.util.concurrent.ListenableFuture;
	
	import java.util.UUID;
	
	/**
	 * 生产者
	 * 使用@EnableScheduling注解开启定时任务
	 */
	@Component
	@EnableScheduling
	public class KafkaProducer {
	
	    @Autowired
	    private KafkaTemplate kafkaTemplate;
	
	    /**
	     * 定时任务1
	     */
	    @Scheduled(cron = "00/1 * * * * ?")
	    public void send(){
	        String message = UUID.randomUUID().toString();
	        ListenableFuture future = kafkaTemplate.send("test", message);
	        future.addCallback(o -> System.out.println("消息发送成功：" + message), throwable -> System.out.println("消息发送失败：" + message));
	    }
	}


使用 kafkaTemplate 发送消息

### SpringbootKafkaProducerApplication

	package com.wzy.springboot_kafka_producer;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootKafkaProducerApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootKafkaProducerApplication.class, args);
	    }
	
	}



### application.yml

	spring:
	  profiles:
	    active: dev #选择要用那个配置文件


### application-dev.yml


	server:
	  port: 8888
	spring:
	  kafka:
	    producer:
	      bootstrap-servers: 47.112.142.231:9092 #服务器ip+端口


### pom.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.2.1.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>springboot_kafka_producer</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_kafka_producer</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter</artifactId>
	        </dependency>
	
	        <dependency>
	            <groupId>org.springframework.kafka</groupId>
	            <artifactId>spring-kafka</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-web</artifactId>
	        </dependency>
	
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-test</artifactId>
	            <scope>test</scope>
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



## Comsumer

![](../Images/5.png)


### public class KafkaConsumer 


	package com.wzy.springboot_kafka_cusumer.consumer;
	
	import org.springframework.kafka.annotation.KafkaListener;
	import org.springframework.stereotype.Component;
	
	/**
	 * 消费者
	 * 使用@KafkaListener注解,可以指定:主题,分区,消费组
	 */
	@Component
	public class KafkaConsumer {
	
	    @KafkaListener(topics = {"test"})
	    public void receive(String message){
	        System.out.println("test--消费消息:" + message);
	    }
	}



### SpringbootKafkaCusumerApplication

	package com.wzy.springboot_kafka_cusumer;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootKafkaCusumerApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootKafkaCusumerApplication.class, args);
	    }
	
	}


### application.yml

	server:
	  port: 8082
	spring:
	  kafka:
	    consumer:
	      group-id: test
	      bootstrap-servers: 47.112.142.231:9092,/47.112.142.231:9093,/47.112.142.231:9094


### pom.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.2.1.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>springboot_kafka_cusumer</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_kafka_cusumer</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter</artifactId>
	        </dependency>
	
	        <dependency>
	            <groupId>org.springframework.kafka</groupId>
	            <artifactId>spring-kafka</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-web</artifactId>
	        </dependency>
	
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-test</artifactId>
	            <scope>test</scope>
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



## 验证

启动producer项目

![](../Images/6.png)


启动consumer项目

![](../Images/7.png)