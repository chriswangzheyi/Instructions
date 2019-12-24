# Springboot 事务管理

---

## 实现方式

使用@Transactional注解来申明事务

配合使用注解

	<dependency>
		<groupId>org.springframework.boot</groupId>
		<artifactId>spring-boot-starter-jdbc</artifactId>
	</dependency>

## 项目结构

![](../Images/1.png)

整合了Mybatis

## 准备工作

在Mysql中，在test数据库中创建表：

	DROP TABLE IF EXISTS tbl_account;
	CREATE TABLE tbl_account (
	  id int(11) NOT NULL AUTO_INCREMENT,
	  name varchar(20) NOT NULL,
	  balance float,
	  PRIMARY KEY (id)
	) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8;
	
	
	insert into tbl_account(id,name,balance) values(1, 'andy','200');
	insert into tbl_account(id,name,balance) values(2, 'lucy','300');


![](../Images/2.png)



## 代码

### AccountController

	package com.wzy.springboot_transactional.controller;
	
	import com.wzy.springboot_transactional.service.AccountService;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.web.bind.annotation.RequestMapping;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	@RequestMapping(value = "/account")
	public class AccountController {
	
	    @Autowired
	    private AccountService accountService;
	
	
	    @RequestMapping("/transfer")
	    public String test(){
	        try {
	            // andy 给lucy转账50元
	            accountService.transfer(1, 2, 50);
	            return "转账成功";
	        } catch (Exception e) {
	            e.printStackTrace();
	            return "转账失败";
	        }
	    }
	}


### AccountDao

	package com.wzy.springboot_transactional.dao;
	
	import org.apache.ibatis.annotations.Param;
	
	public interface AccountDao {
	    public void moveIn(@Param("id") int id, @Param("money") float money); // 转入
	
	    public void moveOut(@Param("id") int id, @Param("money") float money); // 转出
	
	}

### AccountServiceImpl

	package com.wzy.springboot_transactional.impl;
	
	import com.wzy.springboot_transactional.dao.AccountDao;
	import com.wzy.springboot_transactional.service.AccountService;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.stereotype.Service;
	
	@Service
	public class AccountServiceImpl implements AccountService {
	
	    @Autowired(required=false)
	    private AccountDao accountDao;
	
	    // 放开注解后，改方法实现事务控制
	    //@Transactional
	    public void transfer(int outter, int inner, Integer money) {
	
	        accountDao.moveOut(outter, money); //转出
	
	        //int i = 1/0;  // 抛出异常，模拟程序出错
	
	        accountDao.moveIn(inner, money); //转入
	
	    }
	}

此时，不开启@Transactional注解，不抛出错误

### Account

	package com.wzy.springboot_transactional.pojo;
	
	public class Account {
	    private int id;
	    private String name;
	    private float balance;
	
	    public int getId() {
	        return id;
	    }
	
	    public void setId(int id) {
	        this.id = id;
	    }
	
	    public String getName() {
	        return name;
	    }
	
	    public void setName(String name) {
	        this.name = name;
	    }
	
	    public float getBalance() {
	        return balance;
	    }
	
	    public void setBalance(float balance) {
	        this.balance = balance;
	    }
	}


### AccountService

	package com.wzy.springboot_transactional.service;
	
	import org.springframework.stereotype.Service;
	
	@Service
	public interface AccountService {
	
	    //转账
	    public void transfer(int outter,int inner,Integer money);
	}

### SpringbootTransactionalApplication

	package com.wzy.springboot_transactional;
	
	import org.mybatis.spring.annotation.MapperScan;
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	@MapperScan("com.wzy.springboot_transactional.dao")
	public class SpringbootTransactionalApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootTransactionalApplication.class, args);
	    }
	
	}

注意这里需要MapperScan, 指明DAO的位置,添加注释@EnableAutoConfiguration。

### account.xml

	<?xml version="1.0" encoding="UTF-8" ?>
	<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd" >
	
	<mapper namespace="com.wzy.springboot_transactional.dao.AccountDao">
	
	    <!-- 转入 -->
	    <update id="moveIn" >
	        update tbl_account
	        set balance = balance + #{money }
	        where id= #{id,jdbcType=INTEGER}
	    </update>
	
	    <!-- 转出 -->
	    <update id="moveOut" >
	        update tbl_account
	        set balance = balance - #{money }
	        where id= #{id,jdbcType=INTEGER}
	    </update>
	
	</mapper>


### application.yml

	spring:
	  datasource:
	    username: root
	    password: root
	    driver-class-name: com.mysql.jdbc.Driver
	    url: jdbc:mysql://47.112.142.231:3306/test?useUnicode=true&characterEncoding=utf-8&useSSL=false
	
	mybatis:
	  type-aliases-package: com.wzy.springboot_transactional.pojo
	  mapper-locations: classpath:mapper/*.xml


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
	    <artifactId>springboot_transactional</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_transactional</name>
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
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-web</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>mysql</groupId>
	            <artifactId>mysql-connector-java</artifactId>
	            <version>5.1.47</version>
	        </dependency>
	       <!-- 事务管理的核心包-->
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-jdbc</artifactId>
	        </dependency>
	        <!-- Junit依赖 -->
	        <dependency>
	            <groupId>junit</groupId>
	            <artifactId>junit</artifactId>
	            <scope>test</scope>
	        </dependency>
	
	        <dependency>
	            <groupId>org.mybatis.spring.boot</groupId>
	            <artifactId>mybatis-spring-boot-starter</artifactId>
	            <version>2.1.1</version>
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

浏览器测试 http://localhost:8080/account/transfer , 测试显示 转账成功，看看数据库的数据，andy余额是150， lucy余额350，都是对的，如下图所示。

![](../Images/3.png)



## 抛出错误的时候


把AccountServiceImpl文件的 int i = 1/0;  解除注释

### AccountServiceImpl

	package com.wzy.springboot_transactional.impl;
	
	import com.wzy.springboot_transactional.dao.AccountDao;
	import com.wzy.springboot_transactional.service.AccountService;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.stereotype.Service;
	
	@Service
	public class AccountServiceImpl implements AccountService {
	
	    @Autowired(required=false)
	    private AccountDao accountDao;
	
	    // 放开注解后，改方法实现事务控制
	    //@Transactional
	    public void transfer(int outter, int inner, Integer money) {
	
	        accountDao.moveOut(outter, money); //转出
	
	        int i = 1/0;  // 抛出异常，模拟程序出错
	
	        accountDao.moveIn(inner, money); //转入
	
	    }
	}


把数据库的数据恢复成最初的 andy-200, lucy-300, 然后启动类测试，浏览器输入 http://localhost:8080/account/transfer , 测试显示 转账失败，看看数据库的数据，andy余额是150， lucy余额300，如下图所示。

![](../Images/4.png)

andy 转出成功，  lucy转入失败

相当于转出成功，转入没有成功，这是不对的，应该都成功，或者都不成功


## 加上事务管理

在方法上加上@Transactional注解

### AccountServiceImpl

	package com.wzy.springboot_transactional.impl;
	
	import com.wzy.springboot_transactional.dao.AccountDao;
	import com.wzy.springboot_transactional.service.AccountService;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.stereotype.Service;
	
	@Service
	public class AccountServiceImpl implements AccountService {
	
	    @Autowired(required=false)
	    private AccountDao accountDao;
	
	    @Transactional
	    public void transfer(int outter, int inner, Integer money) {
	
	        accountDao.moveOut(outter, money); //转出
	
	        int i = 1/0;  // 抛出异常，模拟程序出错
	
	        accountDao.moveIn(inner, money); //转入
	
	    }
	}

再把数据库的数据恢复成最初的 andy-200, lucy-300, 然后启动类测试，浏览器输入 http://localhost:8080/account/transfer , 测试显示 转账失败，看看数据库的数据，andy余额是200， lucy余额300，如下图所示。

![](../Images/5.png)