# Springboot集成Swagger


## 项目结构

![](../Images/1.png)


## 代码

###SwaggerConfig

	package com.wzy.springboot_swagger;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootSwaggerApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootSwaggerApplication.class, args);
	    }
	
	}


### TestController

	package com.wzy.springboot_swagger.controller;
	
	import org.springframework.web.bind.annotation.GetMapping;
	import org.springframework.web.bind.annotation.PostMapping;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	public class TestController {
	
	    @GetMapping("/hello")
	    private  String hello(){
	        return "hello";
	    }
	
	    @PostMapping("/input")
	    private String input(String info){
	        return info;
	    }
	}


### SpringbootSwaggerApplication

	package com.wzy.springboot_swagger;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootSwaggerApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootSwaggerApplication.class, args);
	    }
	
	}



### POM.xml


	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.3.0.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>springboot_swagger</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_swagger</name>
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
	            <exclusions>
	                <exclusion>
	                    <groupId>org.junit.vintage</groupId>
	                    <artifactId>junit-vintage-engine</artifactId>
	                </exclusion>
	            </exclusions>
	        </dependency>
	
	<!--        swagger相关配置-->
	        <dependency>
	            <groupId>io.springfox</groupId>
	            <artifactId>springfox-swagger-ui</artifactId>
	            <version>2.9.2</version>
	        </dependency>
	
	        <dependency>
	            <groupId>io.springfox</groupId>
	            <artifactId>springfox-swagger2</artifactId>
	            <version>2.9.2</version>
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



## UI管理界面

	http://localhost:8080/swagger-ui.html