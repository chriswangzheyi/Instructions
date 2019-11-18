# Springboot 拦截器 Demo

## 项目结构



![](../Images/1.png)


### LoginInterceptor

编写构造器，继承HandlerInterceptor， 重写preHandle，postHandle， afterCompletion 方法


	package com.wzy.springboot_interceptor.interceptor;
	
	import org.springframework.web.servlet.HandlerInterceptor;
	import org.springframework.web.servlet.ModelAndView;
	
	import javax.servlet.http.HttpServletRequest;
	import javax.servlet.http.HttpServletResponse;
	
	public class LoginInterceptor implements HandlerInterceptor {
	
	    @Override
	    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
	        System.out.println("已经进入拦截器...");
	        return false;
            // return false: 不执行postHandle和afterCompletion。拦截器卡住
            // return ture: 执行postHandle和afterCompletion
	    }
	
	    @Override
	    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView) throws Exception {
	        System.out.println("请求处理之后进行调用");
	    }
	
	    @Override
	    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) throws Exception {
	        System.out.println("整个请求结束之后被调用");
	
	    }
	}



### WebConfig

编写配置文件。

addPathPatterns：请求被拦截

excludePathPatterns：请求不被拦截


	package com.wzy.springboot_interceptor.config;
	
	import com.wzy.springboot_interceptor.interceptor.LoginInterceptor;
	import org.springframework.context.annotation.Configuration;
	import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
	import org.springframework.web.servlet.config.annotation.WebMvcConfigurationSupport;
	
	@Configuration
	public class WebConfig extends WebMvcConfigurationSupport {
	
	
	    @Override
	    protected void addInterceptors(InterceptorRegistry registry) {
	        registry.addInterceptor(new LoginInterceptor())
	                .addPathPatterns("/**")
	                .excludePathPatterns("/login");
	        //这里的配置意思是会拦截所有的请求，除了login的。拦截的请求会根据LoginInterceptor所写逻辑处理
	    }
	}


### LoginController

	package com.wzy.springboot_interceptor.controller;
	
	import org.springframework.web.bind.annotation.GetMapping;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	public class LoginController {
	
	        @GetMapping(value = "/login")
	        public String login(){
	            return "login request not intercept";
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
	        <version>2.2.1.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>springboot_interceptor</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_interceptor</name>
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

请求login：
	
	http://localhost:8080/login 

页面显示：


![](../Images/2.png)


请求login以外的：

	http://localhost:8080/login1

页面报错

界面输出：

	已经进入拦截器...
	请求处理之后进行调用
	整个请求结束之后被调用


 