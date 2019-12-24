# Springboot AOP

---

## 概述

Spring AOP是一个对AOP原理的一种实现方式，另外还有其他的AOP实现如AspectJ等。

AOP意为面向切面编程，是通过预编译方式和运行期动态代理实现程序功能的统一维护的一种技术，是OOP面向对象编程的一种补足。它是软件开发中的一个热点技术，Spring AOP 也是Spring框架的核心特性之一（另一个核心特性是IOC）。

通过AOP技术，我们希望实现一种通用逻辑的解耦，解决一些系统层面上的问题，如日志、事务、权限等，从而提高应用的可重用性和可维护性，和开发效率。

AOP中包括 5 大核心概念：切面（Aspect）、连接点（JoinPoint）、通知（Advice）、切入点（Pointcut）、AOP代理（Proxy）。（记忆口诀：通知 代理 厨师两点（连接点、切入点）切面包。）

### 五大通知类型
1、前置通知 [ Before advice ] ：在连接点前面执行，前置通知不会影响连接点的执行，除非此处抛出异常；

2、正常返回通知 [ After returning advice ] ：在连接点正常执行完成后执行，如果连接点抛出异常，则不会执行；

3、异常返回通知 [ After throwing advice ] ：在连接点抛出异常后执行；

4、返回通知 [ After (finally) advice ] ：在连接点执行完成后执行，不管正常执行完成，还是抛出异常，都会执行返回通知中的内容；

5、环绕通知 [ Around advice ] ：环绕通知围绕在连接点前后，比如一个方法调用的前后。这种通知是最强大的通知，能在方法调用前后自定义一些操作。


## 应用案例分析
在OOP中的基本单元是类，而在AOP中的基本单元是Aspect，它实际上也是一个类，只不过这个类用于管理一些具体的通知方法和切入点。

所谓的连接点，实际上就是一个具体的业务方法，比如Controller中的一个请求方法，而切入点则是带有通知的连接点，在程序中主要体现为书写切入点表达式，这个表达式将会定义一个连接点。

就以Controller中的一个请求方法为例，通过AOP的方式实现一定的业务逻辑。

这个逻辑是：GET请求某一方法，然后通过一个Aspect来实现在这个方法调用前和调用后做一些日志输出处理。

## 代码

![](../Images/2.png)


### DoHomeWorkAspect

	package com.wz.springboot_aop.Aspect;
	
	import org.aspectj.lang.ProceedingJoinPoint;
	import org.aspectj.lang.annotation.Around;
	import org.aspectj.lang.annotation.Aspect;
	import org.aspectj.lang.annotation.Before;
	import org.aspectj.lang.annotation.Pointcut;
	import org.springframework.stereotype.Component;
	import org.springframework.web.context.request.RequestContextHolder;
	import org.springframework.web.context.request.ServletRequestAttributes;
	
	import javax.servlet.http.HttpServletRequest;

	@Aspect
	@Component
	public class DoHomeWorkAspect {
	    /** 定义切入点 */
	    @Pointcut("execution(* com.wz.springboot_aop.controller.DoHomeWorkController.doHomeWork(..))")
	    public void homeWorkPointcut() {
	    }
	
	    /** 定义Before advice通知类型处理方法 */
	    @Before("homeWorkPointcut()")
	    public void beforeHomeWork() {
	        ServletRequestAttributes requestAttributes = (ServletRequestAttributes) RequestContextHolder
	                .getRequestAttributes();
	        HttpServletRequest request = requestAttributes.getRequest();
	        System.out.println(request.getParameter("name") + "想先吃个冰淇淋......");
	    }
	
	}


### DoHomeWorkController

	package com.wz.springboot_aop.controller;
	
	import org.springframework.web.bind.annotation.GetMapping;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	public class DoHomeWorkController {
	    @GetMapping("/dohomework")
	    public void doHomeWork(String name) {
	        System.out.println(name + "做作业... ...");
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
	    <groupId>com.wz</groupId>
	    <artifactId>springboot_aop</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_aop</name>
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


关键是需要引入aop依赖


## 验证

访问：

	http://localhost:8080/dohomework?name=test

![](../Images/3.png)


## 加入@Around注解

### DoHomeWorkAspect

	package com.wz.springboot_aop.Aspect;
	
	import org.aspectj.lang.ProceedingJoinPoint;
	import org.aspectj.lang.annotation.Around;
	import org.aspectj.lang.annotation.Aspect;
	import org.aspectj.lang.annotation.Before;
	import org.aspectj.lang.annotation.Pointcut;
	import org.springframework.stereotype.Component;
	import org.springframework.web.context.request.RequestContextHolder;
	import org.springframework.web.context.request.ServletRequestAttributes;
	
	import javax.servlet.http.HttpServletRequest;
	
	@Aspect
	@Component
	public class DoHomeWorkAspect {
	    /** 定义切入点 */
	    @Pointcut("execution(* com.wz.springboot_aop.controller.DoHomeWorkController.doHomeWork(..))")
	    public void homeWorkPointcut() {
	    }
	
	    /** 定义Before advice通知类型处理方法 */
	    @Before("homeWorkPointcut()")
	    public void beforeHomeWork() {
	        ServletRequestAttributes requestAttributes = (ServletRequestAttributes) RequestContextHolder
	                .getRequestAttributes();
	        HttpServletRequest request = requestAttributes.getRequest();
	        System.out.println(request.getParameter("name") + "想先吃个冰淇淋......");
	    }
	
	    /** 定义方法前后的处理方法 */
	    @Around("homeWorkPointcut()")
	    public void around(ProceedingJoinPoint joinPoint) {
	        System.out.println("环绕通知，方法执行前");
	        try {
	            joinPoint.proceed();
	        } catch (Throwable e) {
	            e.printStackTrace();
	        }
	        System.out.println("环绕通知，方法执行后");
	    }
	}


## 验证

访问

	http://localhost:8080/dohomework?name=test

![](../Images/4.png)

可以看到Round是在Before前执行的


## 访问先后关系

![](../Images/1.png)