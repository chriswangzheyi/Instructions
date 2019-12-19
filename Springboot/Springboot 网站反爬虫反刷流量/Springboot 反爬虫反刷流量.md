# Springboot 反爬虫反刷流量

## 原理

###Redis端：

利用Redis记录请求者的ip地址作为主键，设置一个过期时间。每请求增加一次记录，在过期时间内如果超过某一个上线，则在Redis中增加至黑名单。


###Springboot端：

利用拦截器，在跟系统交互之前，先跟redis交互。


![](../Images/1.png)


## 代码

![](../Images/2.png)


### Myconfig

	package com.wzy.springboot_anticrawler.config;
	
	import com.wzy.springboot_anticrawler.interceptor.MyInterceptor;
	import org.springframework.context.annotation.Configuration;
	import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
	import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
	
	import javax.annotation.Resource;
	
	@Configuration
	public class Myconfig implements WebMvcConfigurer {
	
	    @Resource
	    private MyInterceptor myInterceptor;
	
	
	    @Override
	    public void addInterceptors(InterceptorRegistry registry) {
	        registry.addInterceptor(myInterceptor).addPathPatterns("/hi");
	    }
	}


### testController

	package com.wzy.springboot_anticrawler.controller;
	
	import org.springframework.web.bind.annotation.GetMapping;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	public class testController {
	
	    @GetMapping("/hi")
	    public String test(){
	        return "success";
	    }
	
	}


### MyInterceptor

	package com.wzy.springboot_anticrawler.interceptor;
	
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.data.redis.core.RedisTemplate;
	import org.springframework.stereotype.Component;
	import org.springframework.web.servlet.HandlerInterceptor;
	
	import javax.servlet.http.HttpServletRequest;
	import javax.servlet.http.HttpServletResponse;
	import java.util.concurrent.TimeUnit;
	import org.apache.commons.codec.digest.DigestUtils;
	
	@Component
	public class MyInterceptor implements HandlerInterceptor {
	
	    @Autowired
	    private RedisTemplate redisTemplate;
	
	    @Override
	    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
	        String clientIP = request.getRemoteAddr();
	        String userAgent = request.getHeader("User-Agent");
	        String key = "anti:refresh:" + DigestUtils.md5Hex(clientIP + "_" + userAgent);
	        response.setContentType("text/html;charset=utf-8");
	
	        if(redisTemplate.hasKey("anti:refresh:blacklist")){
	            if (redisTemplate.opsForSet().isMember("anti:refresh:blacklist", clientIP)) {
	                response.getWriter().println("检测到您的IP访问异常，已被加入黑名单");
	                System.out.println("检测到您的IP访问异常，已被加入黑名单");
	                return false;
	            }
	        }
	
	        //计数器
	        Object keyNum =redisTemplate.opsForValue().get(key) ;
	        Integer num = null;
	        if(keyNum != null){
	            num = Integer.valueOf(String.valueOf(keyNum));
	        }
	
	        if(num == null){ //第一次访问
	            redisTemplate.opsForValue().set(key, String.valueOf(1), 60, TimeUnit.SECONDS);
	        }else{
	
	            if(num > 30 && num  < 100){
	                response.getWriter().println("请求过于频繁，请稍后再试!");
	                System.out.println("请求过于频繁，请稍后再试!");
	                redisTemplate.opsForValue().increment(key, 1);
	                return false;
	            }else if(num >=100){
	                response.getWriter().println("检测到您的IP访问异常，已被加入黑名单");
	                System.out.println("检测到您的IP访问异常，已被加入黑名单");
	                redisTemplate.opsForSet().add("anti:refresh:blacklist" , clientIP);
	                return false;
	            }else{
	                redisTemplate.opsForValue().increment(key, 1);
	            }
	        }
	        return true;
	    }
	}



### RedisConfig

	package com.wzy.springboot_anticrawler.utils;
	
	import org.springframework.cache.interceptor.KeyGenerator;
	import org.springframework.context.annotation.Bean;
	import org.springframework.context.annotation.Configuration;
	import org.springframework.data.redis.connection.RedisConnectionFactory;
	import org.springframework.data.redis.core.RedisTemplate;
	import org.springframework.data.redis.serializer.Jackson2JsonRedisSerializer;
	import org.springframework.data.redis.serializer.RedisSerializer;
	import org.springframework.data.redis.serializer.StringRedisSerializer;
	
	import java.lang.reflect.Method;
	
	
	@Configuration
	public class RedisConfig {
	
	    //避免生成乱码
	    @Bean
	    public KeyGenerator keyGenerator() {
	        return new KeyGenerator() {
	            @Override
	            public Object generate(Object target, Method method, Object... params) {
	                StringBuilder sb = new StringBuilder();
	                sb.append(target.getClass().getName());
	                sb.append(method.getName());
	                for (Object obj : params) {
	                    sb.append(obj.toString());
	                }
	                return sb.toString();
	            }
	        };
	    }
	    
	    @Bean
	    public RedisTemplate<String, String> redisTemplate(RedisConnectionFactory redisConnectionFactory) {
	        RedisTemplate<String, String> redisTemplate = new RedisTemplate<>();
	        RedisSerializer stringSerializer = new StringRedisSerializer();
	
	
	        //设置序列化方式
	        redisTemplate.setKeySerializer(stringSerializer);
	        redisTemplate.setValueSerializer(stringSerializer);
	        redisTemplate.setHashKeySerializer(stringSerializer);
	        redisTemplate.setHashValueSerializer(stringSerializer);
	
	        redisTemplate.setConnectionFactory(redisConnectionFactory);
	        return redisTemplate;
	    }
	}


### SpringbootAnticrawlerApplication

	package com.wzy.springboot_anticrawler;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootAnticrawlerApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootAnticrawlerApplication.class, args);
	    }
	
	}


### application.yml

	server:
	  port: 8081
	
	spring:
	  redis:
	    host: 47.112.142.231
	    port: 6379
	    database: 5



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
	    <artifactId>springboot_anticrawler</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_anticrawler</name>
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
	            <artifactId>spring-boot-starter-data-redis</artifactId>
	        </dependency>
	        <!-- spring session redis存储模块 -->
	        <dependency>
	            <groupId>org.springframework.session</groupId>
	            <artifactId>spring-session-data-redis</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-web</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>commons-codec</groupId>
	            <artifactId>commons-codec</artifactId>
	            <version>1.11</version>
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

使用Jmeter请求

![](../Images/3.png)


![](../Images/4.png)




