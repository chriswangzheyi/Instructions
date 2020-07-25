# Springboot集成Hbase


## 在windows运行的时候

需要增加ip 映射

	47.112.142.231 wangzheyi 


## 项目结构

![](../Images/4.png)

## 代码

### HbaseConfig

	package com.wzy.springboot_hbase.config;
	
	import org.apache.hadoop.hbase.HBaseConfiguration;
	import org.springframework.boot.context.properties.EnableConfigurationProperties;
	import org.springframework.context.annotation.Bean;
	import org.springframework.context.annotation.Configuration;
	
	import java.util.Map;
	import java.util.Set;
	
	@Configuration
	@EnableConfigurationProperties(HbaseProperties.class)
	public class HbaseConfig {
	
	    private final HbaseProperties properties;
	
	    public HbaseConfig(HbaseProperties properties) {
	        this.properties = properties;
	    }
	
	    @Bean
	    public org.apache.hadoop.conf.Configuration configuration() {
	
	        org.apache.hadoop.conf.Configuration configuration = HBaseConfiguration.create();
	
	        Map<String, String> config = properties.getConfig();
	
	        Set<String> keySet = config.keySet();
	        for (String key : keySet) {
	            configuration.set(key, config.get(key));
	        }
	        return configuration;
	    }
	
	}


### HbaseProperties

	package com.wzy.springboot_hbase.config;
	
	import lombok.Data;
	import org.springframework.boot.context.properties.ConfigurationProperties;
	
	import java.util.Map;
	
	@Data
	@ConfigurationProperties(prefix = "hbase")
	public class HbaseProperties {
	
	    private Map<String, String> config;
	} 

### SpringbootHbaseApplication

	package com.wzy.springboot_hbase;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootHbaseApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootHbaseApplication.class, args);
	    }
	
	}


### application.yml

	## HBase 配置
	hbase:
	  config:
	    hbase.zookeeper.quorum: 47.112.142.231
	    hbase.zookeeper.port: 2181
	    hbase.zookeeper.znode: /hbase
	    hbase.client.keyvalue.maxsize: 1572864000

### SpringbootHbaseApplicationTests

	package com.wzy.springboot_hbase;
	
	import com.wzy.springboot_hbase.utils.HBaseClient;
	import org.junit.jupiter.api.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.boot.test.context.SpringBootTest;
	
	import java.io.IOException;
	
	@SpringBootTest
	class SpringbootHbaseApplicationTests {
	
	    @Autowired
	    HBaseClient client;
	
	    @Test
	    void contextLoads() throws IOException {
	
	        //删除namespace
	        client.deleteTable("ns1");
	
	        //创建namespace
	        client.createTable("ns1","test","cf");
	
	        //判断是否存在某个namespace
	        Boolean flag = client.tableExists("ns1");
	        System.out.println(flag);
	
	        //插入数据
	        client.insertOrUpdate("ns1","test","cf","a","111");
	
	        //获得数据
	        String val= client.getValue("ns1","test","cf","a");
	        System.out.println(val);
	
	    }
	
	}

### pom.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.3.2.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>springboot_hbase</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_hbase</name>
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
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-configuration-processor</artifactId>
	            <optional>true</optional>
	        </dependency>
	
	
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-test</artifactId>
	            <scope>test</scope>
	            <exclusions>
	                <exclusion>
	                    <groupId>org.junit.vintage</groupId>
	                    <artifactId>junit-vintage-engine</artifactId>
	                </exclusion>
	            </exclusions>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hbase</groupId>
	            <artifactId>hbase-client</artifactId>
	            <version>2.1.1</version>
	            <exclusions>
	                <exclusion>
	                    <groupId>javax.servlet</groupId>
	                    <artifactId>servlet-api</artifactId>
	                </exclusion>
	            </exclusions>
	        </dependency>
	        <dependency>
	            <groupId>org.projectlombok</groupId>
	            <artifactId>lombok</artifactId>
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



注意：

**注意jar包版本和服务器的对应关系
这里服务器版本是2.1.3，jar包是2.1.1