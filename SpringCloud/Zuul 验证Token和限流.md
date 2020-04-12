# Zuul 验证Token 和 限流

---

## 验证Token 


通过 extends ZuulFilter类实现。


自动overwrite方法：

    @Override
    public String filterType() {
        return null;
    }

    @Override
    public int filterOrder() {
        return 0;
    }

    @Override
    public boolean shouldFilter() {
        return false;
    }

    @Override
    public Object run() throws ZuulException {
        return null;
    }


### filterType

分为4种：ERROR_TYPE，POST_TYPE，PRE_TYPE， ROUTE_TYPE。 定义了Filter的类型。  　

### filterOrder

数字越小，执行优先级越高

### shouldFilter

编写例外情况。例外情况满足则return false,否则就return true。

### run

执行鉴权等相关操作。



## Demo项目结构

![](../Images/1.png)



### TokenFilter

	package com.wzy.gateway;
	
	import com.netflix.zuul.ZuulFilter;
	import com.netflix.zuul.context.RequestContext;
	import com.netflix.zuul.exception.ZuulException;
	import org.springframework.stereotype.Component;
	
	import javax.servlet.http.HttpServletRequest;
	
	import static org.springframework.cloud.netflix.zuul.filters.support.FilterConstants.PRE_TYPE;
	
	@Component
	public class TokenFilter extends ZuulFilter{
	
	    @Override
	    public String filterType() {
	        return PRE_TYPE;
	    }
	
	    @Override
	    public int filterOrder() {
	        return 1;
	    }
	
	    @Override
	    public boolean shouldFilter() {
	            return true;
	    }
	
	    @Override
	    public Object run() throws ZuulException {
	
	        RequestContext ctx = RequestContext.getCurrentContext();
	        HttpServletRequest request = ctx.getRequest();
	
	        Object token = request.getParameter("token");
	
	        //校验token
	        if (token == null) {
	            ctx.setSendZuulResponse(false);
	            ctx.setResponseStatusCode(401);
	            return null;
	        } else {
	            //TODO 根据token获取相应的登录信息，进行校验（略）
	        }
	
	        return null;
	    }
	
	}
	

### GatewayApplication

	package com.wzy.gateway;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
	import org.springframework.cloud.netflix.zuul.EnableZuulProxy;
	
	@SpringBootApplication
	@EnableZuulProxy
	@EnableDiscoveryClient
	public class GatewayApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(GatewayApplication.class, args);
	    }
	
	}


### application.yml

	server:
	  port: 5001
	
	eureka:
	  client:
	    service-url:
	      defaultZone: http://eureka6001:6001/eureka, http://eureka6002:6002/eureka
	  instance:
	    instance-id: gateway   #在信息列表显示主机名称
	    prefer-ip-address: true  # 访问路径变为ip地址
	
	spring:
	  application:
	    name: gateway
	zuul:
	  prefix: /
	  ignored-services:
	    "*"
	  routes:
	    provider: /p/**    #把所有请求provider服务的请求都映射到/p下


### pom.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.2.6.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>gateway</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>gateway</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	        <spring-cloud.version>Hoxton.SR3</spring-cloud.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.cloud</groupId>
	            <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.cloud</groupId>
	            <artifactId>spring-cloud-starter-netflix-zuul</artifactId>
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
	
	    <dependencyManagement>
	        <dependencies>
	            <dependency>
	                <groupId>org.springframework.cloud</groupId>
	                <artifactId>spring-cloud-dependencies</artifactId>
	                <version>${spring-cloud.version}</version>
	                <type>pom</type>
	                <scope>import</scope>
	            </dependency>
	        </dependencies>
	    </dependencyManagement>
	
	    <build>
	        <plugins>
	            <plugin>
	                <groupId>org.springframework.boot</groupId>
	                <artifactId>spring-boot-maven-plugin</artifactId>
	            </plugin>
	        </plugins>
	    </build>
	
	</project>


## 整合

借用最简Demo，项目结构为：

![](../Images/2.png)


### eureka：

![](../Images/3.png)


#### EurekaApplication

	package com.wzy.eureka;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	import org.springframework.cloud.netflix.eureka.server.EnableEurekaServer;
	
	@SpringBootApplication
	@EnableEurekaServer
	public class EurekaApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(EurekaApplication.class, args);
	    }
	
	}


#### application.yml

	server:
	  port: 6001
	
	spring:
	  application:
	    name: eureka6001
	
	eureka:
	  client:
	    fetch-registry: false
	    register-with-eureka: false
	    service-url:
	      defaultZone: http://eureka6001:6001/eureka, http://eureka6002:6002/eureka
	  environment: dev


#### pom.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.2.6.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>eureka</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>eureka</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	        <spring-cloud.version>Hoxton.SR3</spring-cloud.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.cloud</groupId>
	            <artifactId>spring-cloud-starter-netflix-eureka-server</artifactId>
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
	
	    <dependencyManagement>
	        <dependencies>
	            <dependency>
	                <groupId>org.springframework.cloud</groupId>
	                <artifactId>spring-cloud-dependencies</artifactId>
	                <version>${spring-cloud.version}</version>
	                <type>pom</type>
	                <scope>import</scope>
	            </dependency>
	        </dependencies>
	    </dependencyManagement>
	
	    <build>
	        <finalName>eureka-server</finalName>
	        <plugins>
	            <plugin>    <!-- 该插件的主要功能是进行项目的打包发布处理 -->
	                <groupId>org.springframework.boot</groupId>
	                <artifactId>spring-boot-maven-plugin</artifactId>
	                <configuration> <!-- 设置程序执行的主类 -->
	                    <mainClass>com.wzy.eureka.EurekaApplication</mainClass>
	                </configuration>
	                <executions>
	                    <execution>
	                        <goals>
	                            <goal>repackage</goal>
	                        </goals>
	                    </execution>
	                </executions>
	            </plugin>
	        </plugins>
	    </build>
	
	</project>


### provider1

![](../Images/4.png)


#### configs

	package com.wzy.provider1.config;
	
	import com.netflix.hystrix.contrib.metrics.eventstream.HystrixMetricsStreamServlet;
	import org.springframework.boot.web.servlet.ServletRegistrationBean;
	import org.springframework.context.annotation.Bean;
	import org.springframework.context.annotation.Configuration;
	
	@Configuration
	public class configs {
	
	    //hystrix dashboard相关配置
	    @Bean
	    public ServletRegistrationBean getServlet() {
	        HystrixMetricsStreamServlet streamServlet = new HystrixMetricsStreamServlet();
	        ServletRegistrationBean registrationBean = new ServletRegistrationBean(streamServlet);
	        registrationBean.setLoadOnStartup(1);
	        registrationBean.addUrlMappings("/hystrix.stream");
	        registrationBean.setName("HystrixMetricsStreamServlet");
	        return registrationBean;
	    }
	
	}

#### TestController

	package com.wzy.provider1.controller;
	
	import com.netflix.hystrix.contrib.javanica.annotation.HystrixCommand;
	import org.springframework.web.bind.annotation.GetMapping;
	import org.springframework.web.bind.annotation.PathVariable;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	public class TestController {
	
	    @GetMapping("/test/{info}")
	    @HystrixCommand(fallbackMethod = "getFallBack")
	    public String test(@PathVariable("info") String info){
	        int i = 1/0; //模拟错误
	        return "provider1 returns" + info;
	    }
	
	    //fallback方法的请求报文和返回结构需要跟原方法一致
	    public String getFallBack(String info){
	        return "fallback "+ info;
	    }
	
	}


#### Provider1Application

	package com.wzy.provider1;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	import org.springframework.cloud.client.circuitbreaker.EnableCircuitBreaker;
	import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
	
	@EnableDiscoveryClient
	@EnableCircuitBreaker
	@SpringBootApplication
	public class Provider1Application {
	
	    public static void main(String[] args) {
	        SpringApplication.run(Provider1Application.class, args);
	    }
	
	}


#### application.yml
	
	server:
	  port: 8001
	
	eureka:
	  client:
	    service-url:
	      defaultZone: http://eureka6001:6001/eureka, http://eureka6002:6002/eureka
	  instance:
	    instance-id: provider1   #在信息列表显示主机名称
	    prefer-ip-address: true  # 访问路径变为ip地址
	
	spring:
	  application:
	    name: provider  # 名字相同则认为是同一个服务


#### pom.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.2.6.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>provider1</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>provider1</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	        <spring-cloud.version>Hoxton.SR3</spring-cloud.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-web</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.cloud</groupId>
	            <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.cloud</groupId>
	            <artifactId>spring-cloud-starter-netflix-hystrix</artifactId>
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
	
	    <dependencyManagement>
	        <dependencies>
	            <dependency>
	                <groupId>org.springframework.cloud</groupId>
	                <artifactId>spring-cloud-dependencies</artifactId>
	                <version>${spring-cloud.version}</version>
	                <type>pom</type>
	                <scope>import</scope>
	            </dependency>
	        </dependencies>
	    </dependencyManagement>
	
	    <build>
	        <plugins>
	            <plugin>
	                <groupId>org.springframework.boot</groupId>
	                <artifactId>spring-boot-maven-plugin</artifactId>
	            </plugin>
	        </plugins>
	    </build>
	
	</project>


### gateway

![](../Images/5.png)

#### GatewayApplication

	package com.wzy.gateway;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
	import org.springframework.cloud.netflix.zuul.EnableZuulProxy;
	
	@SpringBootApplication
	@EnableZuulProxy
	@EnableDiscoveryClient
	public class GatewayApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(GatewayApplication.class, args);
	    }
	
	}


#### TokenFilter

	package com.wzy.gateway;
	
	import com.netflix.zuul.ZuulFilter;
	import com.netflix.zuul.context.RequestContext;
	import com.netflix.zuul.exception.ZuulException;
	import org.springframework.stereotype.Component;
	
	import javax.servlet.http.HttpServletRequest;
	
	import static org.springframework.cloud.netflix.zuul.filters.support.FilterConstants.PRE_TYPE;
	
	@Component
	public class TokenFilter extends ZuulFilter{
	
	    @Override
	    public String filterType() {
	        return PRE_TYPE;
	    }
	
	    @Override
	    public int filterOrder() {
	        return 1;
	    }
	
	    @Override
	    public boolean shouldFilter() {
	            return true;
	    }
	
	    @Override
	    public Object run() throws ZuulException {
	
	        RequestContext ctx = RequestContext.getCurrentContext();
	        HttpServletRequest request = ctx.getRequest();
	
	        Object token = request.getParameter("token");
	
	        //校验token
	        if (token == null) {
	            ctx.setSendZuulResponse(false);
	            ctx.setResponseStatusCode(401);
	            return null;
	        } else {
	            //TODO 根据token获取相应的登录信息，进行校验（略）
	        }
	
	        return null;
	    }
	
	}


#### application.yml

	server:
	  port: 5001
	
	eureka:
	  client:
	    service-url:
	      defaultZone: http://eureka6001:6001/eureka, http://eureka6002:6002/eureka
	  instance:
	    instance-id: gateway   #在信息列表显示主机名称
	    prefer-ip-address: true  # 访问路径变为ip地址
	
	spring:
	  application:
	    name: gateway
	zuul:
	  prefix: /
	  ignored-services:
	    "*"
	  routes:
	    provider: /p/**    #把所有请求provider服务的请求都映射到/p下


#### pom.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.2.6.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>gateway</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>gateway</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	        <spring-cloud.version>Hoxton.SR3</spring-cloud.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.cloud</groupId>
	            <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.cloud</groupId>
	            <artifactId>spring-cloud-starter-netflix-zuul</artifactId>
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
	
	    <dependencyManagement>
	        <dependencies>
	            <dependency>
	                <groupId>org.springframework.cloud</groupId>
	                <artifactId>spring-cloud-dependencies</artifactId>
	                <version>${spring-cloud.version}</version>
	                <type>pom</type>
	                <scope>import</scope>
	            </dependency>
	        </dependencies>
	    </dependencyManagement>
	
	    <build>
	        <plugins>
	            <plugin>
	                <groupId>org.springframework.boot</groupId>
	                <artifactId>spring-boot-maven-plugin</artifactId>
	            </plugin>
	        </plugins>
	    </build>
	
	</project>



## 测试


### 请求报文带token

	http://localhost:5001/p/test/11?token=11

可以经过zuul正常访问provider


### 请求报文不带Token

请求不到provider



## 网关限流


### pom

增加：

	<dependency>
		<groupId>com.marcosbarbero.cloud</groupId>
        <artifactId>spring-cloud-zuul-ratelimit</artifactId>
        <version>2.4.0.RELEASE</version>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-redis</artifactId>
    </dependency>


此时的完整pom是：

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.2.6.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>gateway</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>gateway</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	        <spring-cloud.version>Hoxton.SR3</spring-cloud.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.cloud</groupId>
	            <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.cloud</groupId>
	            <artifactId>spring-cloud-starter-netflix-zuul</artifactId>
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
	            <groupId>com.marcosbarbero.cloud</groupId>
	            <artifactId>spring-cloud-zuul-ratelimit</artifactId>
	            <version>2.4.0.RELEASE</version>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-data-redis</artifactId>
	        </dependency>
	
	    </dependencies>
	
	    <dependencyManagement>
	        <dependencies>
	            <dependency>
	                <groupId>org.springframework.cloud</groupId>
	                <artifactId>spring-cloud-dependencies</artifactId>
	                <version>${spring-cloud.version}</version>
	                <type>pom</type>
	                <scope>import</scope>
	            </dependency>
	        </dependencies>
	    </dependencyManagement>
	
	    <build>
	        <plugins>
	            <plugin>
	                <groupId>org.springframework.boot</groupId>
	                <artifactId>spring-boot-maven-plugin</artifactId>
	            </plugin>
	        </plugins>
	    </build>
	
	</project>


### application.yml

增加redis和ratelimit相关内容：

	spring:
	  redis:
	    host: 47.112.142.231
	zuul:
	  ratelimit:
	    key-prefix: zheyi
	    enabled: true
	    repository: REDIS
	    behind-proxy: true
	    default-policy-list: #optional - will apply unless specific policy exists
	      - limit: 1 #optional - request number limit per refresh interval window
	        quota: 1 #optional - request time limit per refresh interval window (in seconds)
	        refresh-interval: 3 #default value (in seconds)


按照配置是每3秒只能请求一次。这里的配置是**全局配置**，对所有的请求都有效。

参考配置： https://github.com/marcosbarbero/spring-cloud-zuul-ratelimit

### 测试：

	http://localhost:5001/p/test/11?token=11

请求过快报错：

![](../Images/6.png)


