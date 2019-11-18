# Springboot 整合 Redis 生成token

## 设计思路

将用户名用MD5加密后作为redis主键，利用redis setnx判断token是否存在以及是否


## 代码

![](../Images/1.png)


### RedisConfig

	package com.wzy.springboot_redis_idempotence.config;
	
	import org.springframework.cache.interceptor.KeyGenerator;
	import org.springframework.context.annotation.Bean;
	import org.springframework.context.annotation.Configuration;
	import org.springframework.data.redis.connection.RedisConnectionFactory;
	import org.springframework.data.redis.core.RedisTemplate;
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

### UserController

	package com.wzy.springboot_redis_idempotence.controller;
	
	import com.alibaba.fastjson.JSONObject;
	
	import com.wzy.springboot_redis_idempotence.pojo.TokenInfo;
	import com.wzy.springboot_redis_idempotence.pojo.User;
	import com.wzy.springboot_redis_idempotence.service.TokenService;
	import com.wzy.springboot_redis_idempotence.service.UserService;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.stereotype.Controller;
	import org.springframework.web.bind.annotation.RequestMapping;
	import org.springframework.web.bind.annotation.ResponseBody;
	
	import javax.servlet.http.HttpServletRequest;
	
	@Controller
	@RequestMapping("/user")
	public class UserController {
	
	    @Autowired
	    private UserService userService;
	
	    @Autowired
	    private TokenService tokenService;
	
	    @RequestMapping("/login")
	    @ResponseBody
	    public String login(String username, String password, HttpServletRequest request) {
	        TokenInfo dto = new TokenInfo();
	        User user = this.userService.login(username, password);
	        if (user != null) {
	            String userAgent = request.getHeader("user-agent");
	            String token = this.tokenService.generateToken(userAgent, username);
	            this.tokenService.save(token, user);
	
	            dto.setIsLogin("true");
	            dto.setToken(token);
	            dto.setTokenCreatedDate(System.currentTimeMillis());
	            dto.setTokenExpiryDate(System.currentTimeMillis() + 2*60*60*1000);
	        } else {
	            dto.setIsLogin("false");
	        }
	        return JSONObject.toJSONString(dto);
	    }
	}

### TokenInfo

	package com.wzy.springboot_redis_idempotence.pojo;
	
	public class TokenInfo {
	
	    private String token;
	
	    private Long tokenCreatedDate;
	
	    private Long tokenExpiryDate;
	
	    private String isLogin;
	
	
	    public String getToken() {
	        return token;
	    }
	
	    public void setToken(String token) {
	        this.token = token;
	    }
	
	    public Long getTokenCreatedDate() {
	        return tokenCreatedDate;
	    }
	
	    public void setTokenCreatedDate(Long tokenCreatedDate) {
	        this.tokenCreatedDate = tokenCreatedDate;
	    }
	
	    public Long getTokenExpiryDate() {
	        return tokenExpiryDate;
	    }
	
	    public void setTokenExpiryDate(Long tokenExpiryDate) {
	        this.tokenExpiryDate = tokenExpiryDate;
	    }
	
	    public String getIsLogin() {
	        return isLogin;
	    }
	
	    public void setIsLogin(String isLogin) {
	        this.isLogin = isLogin;
	    }
	}


### User

	package com.wzy.springboot_redis_idempotence.pojo;
	
	public class User {
	    private Integer id;
	
	    private String username;
	
	    private String password;
	
	    public User(Integer id, String username, String password) {
	        this.id = id;
	        this.username = username;
	        this.password = password;
	    }
	
	    public User() {
	    }
	
	    public Integer getId() {
	        return id;
	    }
	
	    public void setId(Integer id) {
	        this.id = id;
	    }
	
	    public String getUsername() {
	        return username;
	    }
	
	    public void setUsername(String username) {
	        this.username = username;
	    }
	
	    public String getPassword() {
	        return password;
	    }
	
	    public void setPassword(String password) {
	        this.password = password;
	    }
	}

### TokenService

	package com.wzy.springboot_redis_idempotence.service;
	
	import com.alibaba.fastjson.JSONObject;
	
	import com.wzy.springboot_redis_idempotence.pojo.User;
	import com.wzy.springboot_redis_idempotence.utils.RedisUtil;
	import nl.bitwalker.useragentutils.UserAgent;
	import org.apache.commons.codec.digest.DigestUtils;
	import org.springframework.stereotype.Service;
	
	import javax.annotation.Resource;
	import java.text.SimpleDateFormat;
	import java.util.Date;
	import java.util.Random;
	
	@Service("tokenService")
	public class TokenService {
	
	    @Resource
	    private RedisUtil redisUtil;
	
	    //生成token(格式为token:设备-加密的用户名)
	    public String generateToken(String userAgentStr, String username) {
	        StringBuilder token = new StringBuilder("token:");
	        //设备
	        UserAgent userAgent = UserAgent.parseUserAgentString(userAgentStr);
	        if (userAgent.getOperatingSystem().isMobileDevice()) {
	            token.append("MOBILE-");
	        } else {
	            token.append("PC-");
	        }
	        //加密的用户名作为redis的key
	        token.append(DigestUtils.md5Hex(username) + "-");
	        System.out.println("token-->" + token.toString());
	        return token.toString();
	    }
	
	    //把token存到redis中
	    public void save(String token, User user) {
	
	        if (token.startsWith("token:PC")) {
	            redisUtil.setex(token, JSONObject.toJSONString(user), 2*60*60);
	        } else {
	            redisUtil.set(token, JSONObject.toJSONString(user));
	        }
	    }
	}


### UserService

	package com.wzy.springboot_redis_idempotence.service;
	
	import com.wzy.springboot_redis_idempotence.pojo.User;
	import org.springframework.stereotype.Service;
	
	@Service("userService")
	public class UserService {
	    public User login(String username, String password) {
	        if ("tom".equals(username) && "123".equals(password)){
	            return new User(1, "tom", "123");
	        } else {
	            return null;
	        }
	    }
	}


### RedisUtil

	package com.wzy.springboot_redis_idempotence.utils;
	
	import org.springframework.data.redis.core.RedisTemplate;
	import org.springframework.data.redis.core.ValueOperations;
	import org.springframework.stereotype.Component;
	
	import javax.annotation.Resource;
	
	@Component
	public class RedisUtil {
	
	    @Resource
	    private RedisTemplate<String, String> redisTemplate;
	
	    public void set(String key, String value) {
	        ValueOperations<String, String> valueOperations = redisTemplate.opsForValue();
	        valueOperations.set(key, value);
	    }
	
	    public void setex(String key, String value, int seconds) {
	        ValueOperations<String, String> valueOperations = redisTemplate.opsForValue();
	        valueOperations.set(key, value, seconds);
	    }
	}


###　SpringbootRedisIdempotenceApplication

	package com.wzy.springboot_redis_idempotence;
	
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootRedisIdempotenceApplication  {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootRedisIdempotenceApplication.class, args);
	    }

	}

### application.properties

	##指定使用redis数据库索引(默认为0)
	spring.redis.database=0
	##指定Redis服务器地址
	spring.redis.host=47.112.142.231
	##指定Redis端口号
	spring.redis.port=6379
	##指定Redis密码
	spring.redis.password=


###　pom.xml

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
	    <artifactId>springboot_redis_idempotence</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_redis_idempotence</name>
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
	
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-redis</artifactId>
	            <version>1.4.5.RELEASE</version>
	        </dependency>
	
	        <dependency>
	            <groupId>com.alibaba</groupId>
	            <artifactId>fastjson</artifactId>
	            <version>1.2.31</version>
	        </dependency>
	
	        <dependency>
	            <groupId>nl.bitwalker</groupId>
	            <artifactId>UserAgentUtils</artifactId>
	            <version>1.2.4</version>
	        </dependency>
	
	        <dependency>
	            <groupId>commons-codec</groupId>
	            <artifactId>commons-codec</artifactId>
	            <version>1.6</version>
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

使用postman 请求


![](../Images/2.png)


进入到docker 容器中：

	docker exec -it myredis


更换目录:

	cd /usr/local/bin

启动redis：

	redis-cli

查看key:

	keys /




