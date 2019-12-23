# Springboot 集成 Druid

---

## 什么是连接池

数据库连接池负责分配、管理和释放数据库连接，它允许应用程序重复使用一个现有的数据库连接，而不是再重新建立一个。

## 不使用连接池流程

![](../Images/2.png)


	不使用数据库连接池的步骤：

	1.TCP建立连接的三次握手
	2.MySQL认证的三次握手
	3.真正的SQL执行
	4.MySQL的关闭
	5.TCP的四次握手关闭
	可以看到，为了执行一条SQL，却多了非常多我们不关心的网络交互。

## 使用连接池流程

![](../Images/3.png)

第一次访问的时候，需要建立连接。 但是之后的访问，均会复用之前创建的连接，直接执行SQL语句。

## 数据库连接池的工作原理

	连接池的工作原理主要由三部分组成，分别为
	
	1.连接池的建立
	2.连接池中连接的使用管理
	3.连接池的关闭

	        第一、连接池的建立。一般在系统初始化时，连接池会根据系统配置建立，并在池中创建了几个连接对象，以便使用时能从连接池中获取。连接池中的连接不能随意创建和关闭，这样避免了连接随意建立和关闭造成的系统开销。Java中提供了很多容器类可以方便的构建连接池，例如Vector、Stack等。
	
	        第二、连接池的管理。连接池管理策略是连接池机制的核心，连接池内连接的分配和释放对系统的性能有很大的影响。其管理策略是：
	
	        当客户请求数据库连接时，首先查看连接池中是否有空闲连接，如果存在空闲连接，则将连接分配给客户使用；如果没有空闲连接，则查看当前所开的连接数是否已经达到最大连接数，如果没达到就重新创建一个连接给请求的客户；如果达到就按设定的最大等待时间进行等待，如果超出最大等待时间，则抛出异常给客户。
	
	        当客户释放数据库连接时，先判断该连接的引用次数是否超过了规定值，如果超过就从连接池中删除该连接，否则保留为其他客户服务。
	
	        该策略保证了数据库连接的有效复用，避免频繁的建立、释放连接所带来的系统资源开销。
	
	        第三、连接池的关闭。当应用程序退出时，关闭连接池中所有的连接，释放连接池相关的资源，该过程正好与创建相反。


## 项目结构

![](../Images/1.png)


###Configs

	package com.wzy.springboot_druid.config;
	
	import com.alibaba.druid.pool.DruidDataSource;
	import com.alibaba.druid.support.http.StatViewServlet;
	import com.alibaba.druid.support.http.WebStatFilter;
	import org.springframework.boot.context.properties.ConfigurationProperties;
	import org.springframework.boot.web.servlet.FilterRegistrationBean;
	import org.springframework.boot.web.servlet.ServletRegistrationBean;
	import org.springframework.context.annotation.Bean;
	import org.springframework.context.annotation.Configuration;
	
	import javax.sql.DataSource;
	import java.util.HashMap;
	import java.util.Map;
	
	@Configuration
	public class Configs {
	    // 初始化druidDataSource对象
	    @Bean
	    @ConfigurationProperties(prefix = "spring.datasource")
	    public DataSource druidDataSource(){
	        return  new DruidDataSource();
	    }
	
	
	    // 注册后台监控界面
	    @Bean
	    public ServletRegistrationBean servletRegistrationBean(){
	        // 绑定后台监控界面的路径  为localhost/druid
	        ServletRegistrationBean bean=new ServletRegistrationBean(new StatViewServlet(),"/druid/*");
	        Map<String,String>map=new HashMap<>();
	        // 设置后台界面的用户名
	        map.put("loginUsername","admin");
	        //设置后台界面密码
	        map.put("loginPassword","admin");
	        // 设置那些ip允许访问，" " 为所有
	        map.put("allow","");
	        // 不允许该ip访问
	        map.put("deny","33.32.43.123");
	        bean.setInitParameters(map);
	        return bean;
	    }
	
	    // 监听获取应用的数据，filter用于收集数据，servlet用于数据展示
	
	    @Bean
	    public FilterRegistrationBean filterRegistrationBean(){
	        FilterRegistrationBean bean=new FilterRegistrationBean();
	        // 设置过滤器
	        bean.setFilter(new WebStatFilter());
	        // 对所有请求进行过滤捕捉监听
	        bean.addUrlPatterns("/*");
	        Map<String,String> map=new HashMap<>();
	        // 排除 . png  .js 的监听  用于排除一些静态资源访问
	        map.put("exclusions","*.png，*.js");
	        bean.setInitParameters(map);
	        return bean;
	    }
	
	}


### SpringbootDruidApplication

	package com.wzy.springboot_druid;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootDruidApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootDruidApplication.class, args);
	    }
	
	}


### application.yml

	spring:
	  datasource:
	    username: root
	    data-password: root
	    url: jdbc:mysql://47.112.142.231:3306/test?useUnicode=true&characterEncoding=UTF-8
	    driver-class-name: com.mysql.cj.jdbc.Driver
	    type: com.alibaba.druid.pool.DruidDataSource
	    ####### Druid 连接池配置
	    dbType: mysql   # 指定数据库类型 mysql
	    initialSize: 5  # 启动初始化连接数量
	    minIdle: 5      # 最小空闲连接
	    maxActive: 20   # 最大连接数量（包含使用中的和空闲的）
	    maxWait: 60000  # 最大连接等待时间 ，超出时间报错
	    timeBetweenEvictionRunsMillis: 60000  # 设置执行一次连接回收器的时间
	    minEvictableIdleTimeMillis: 300000   # 设置时间： 该时间内没有任何操作的空闲连接会被回收
	    validationQuery: select 'x'         # 验证连接有效性的sql
	    testWhileIdle: true             # 空闲时校验
	    testOnBorrow: false  # 使用中是否校验有效性
	    testOnReturn: false  # 归还连接池时是否校验
	    poolPreparedStatements: false  # mysql 不推荐打开预处理连接池
	    filters: stat,wall,logback  #设置过滤器 stat用于接收状态，wall防止sql注入，logback说明使用logback进行日志输出
	    userGlobalataSourceStat: true  # 统计所有数据源状态
	    connectionProperties: druid.stat.mergSql=true;druid.stat.slowSqlMillis=500  # sql合并统计 设置慢sql时间为500，超过500 会有记录提示
	
	server:
	  port: 8080


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
	    <artifactId>springboot_druid</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_druid</name>
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
	            <groupId>mysql</groupId>
	            <artifactId>mysql-connector-java</artifactId>
	            <scope>runtime</scope>
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
	            <groupId>com.alibaba</groupId>
	            <artifactId>druid</artifactId>
	            <version>1.1.20</version>
	        </dependency>
	        <!-- 如果 不加入这依赖       配置监控统计拦截的filters时   这个会报错 filters: stat,wall,log4j    -->
	        <dependency>
	            <groupId>log4j</groupId>
	            <artifactId>log4j</artifactId>
	            <version>1.2.17</version>
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



## 管理页面

	http://localhost:8080/druid/index.html

账号密码见config中设置。

账号：admin

密码：admin

