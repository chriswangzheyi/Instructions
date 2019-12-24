# Springboot Aop 简单Demo2

---

## 综述

该Demo是验证在Springboot aop下，一个切片对应多个Controller的情况


## 代码

![](../Images/5.png) 


### TestAspect

	package com.wzy.springboot_aop2.Aspect;
	
	import org.aspectj.lang.annotation.After;
	import org.aspectj.lang.annotation.Aspect;
	import org.aspectj.lang.annotation.Before;
	import org.aspectj.lang.annotation.Pointcut;
	import org.springframework.stereotype.Component;
	
	@Component
	@Aspect
	public class TestAspect {
	
	    @Pointcut("execution(public * com.wzy.springboot_aop2.controller.*.*(..))")
	    public void TestPointCut(){
	    }
	
	    @Before("TestPointCut()")
	    public void beforeTest(){
	        System.out.println("进入切片之前");
	    }
	
	    @After("TestPointCut()")
	    public void afterTest(){
	        System.out.println("进入切片之后");
	    }
	}


@Pointcut("execution(public * com.wzy.springboot_aop2.controller.*.*

通过这个注解，将com.wzy.springboot_aop2.controller下所有的文件都包含在切片范围内。


### TestController1

	package com.wzy.springboot_aop2.controller;
	
	import org.springframework.web.bind.annotation.GetMapping;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	public class TestController1 {
	
	    @GetMapping("/test1")
	    public String test1(){
	        System.out.println("111111");
	        return "11";
	    }
	}


### TestController2

	package com.wzy.springboot_aop2.controller;
	
	import org.springframework.web.bind.annotation.GetMapping;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	public class TestController2 {
	
	    @GetMapping("/test2")
	    public String test2(){
	        System.out.println("22222");
	        return "22";
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
	        <version>2.2.2.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>springboot_aop2</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_aop2</name>
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
	        <!--引入spring aop-->
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-aop</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-web</artifactId>
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
	
	http://localhost:8080/test1
	
	http://localhost:8080/test2

分别得到以下的响应：

	进入切片之前
	111111
	进入切片之后
	
	进入切片之前
	22222
	进入切片之后


证明切片成功
