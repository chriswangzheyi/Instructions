**Springboot 集成Redis**


结构


![](../Images/3.PNG)

---

pom:

    <?xml version="1.0" encoding="UTF-8"?>
    <project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.1.1.RELEASE</version>
    <relativePath/> <!-- lookup parent from repository -->
    </parent>
    <groupId>com.wzy</groupId>
    <artifactId>springboot_demo</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>springboot_demo</name>
    <description>Demo project for Spring Boot</description>
       <!-- <packaging>war</packaging> --><!--打包成War用-->
    <packaging>jar</packaging>
    
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
    </dependency>
    
    <!--因配置外部TOMCAT 而配置-->
    <dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-tomcat</artifactId>
    </dependency>
    
    <!--  springboot整合 redis -->
    <dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
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




---

application.yml

    
    server:
      port: 8080
    
    #==========redis 配置信息===========#
    
    spring:
      redis:
    host: 47.112.142.231
    port: 6379
    password:
    jedis:
      pool:
    max-active: 8
    
---

测试类   
SpringbootDemoApplicationTests:


    
    import com.wzy.service.RedisService;
    import org.junit.Test;
    import org.junit.runner.RunWith;
    import org.springframework.beans.factory.annotation.Autowired;
    import org.springframework.boot.test.context.SpringBootTest;
    import org.springframework.test.context.junit4.SpringRunner;
    
    @RunWith(SpringRunner.class)
    @SpringBootTest
    public class SpringbootDemoApplicationTests {
    
    
    @Autowired
    RedisService redisService;
    
    @Test
    public void contextLoads() {
    
    redisService.set("aa","aa");
    
    System.out.println("============================================================="+redisService.get("aa"));
    }
    
    }
    
---

RedisService.java：


    import org.springframework.beans.factory.annotation.Autowired;
    import org.springframework.data.redis.core.RedisTemplate;
    import org.springframework.data.redis.core.ValueOperations;
    import org.springframework.stereotype.Service;
    
    import java.util.concurrent.TimeUnit;
    
    @Service
    public class RedisService <T> {
    
    @Autowired
    private RedisTemplate redisTemplate;
    
    /**
     * 一天有多少分钟，默认时间是一天
     */
    private static final long MINUTES_OF_ONE_DAY = 24 * 60;
    
    
    /**
     * 将 key，value 存放到redis数据库中，默认设置过期时间为一天
     *
     * @param key
     * @param value
     */
    public void set(String key, T value) {
    set(key, value, MINUTES_OF_ONE_DAY);
    }
    
    /**
     * 将 key，value 存放到redis数据库中，设置过期时间单位是分钟
     *
     * @param key
     * @param value
     * @param expireTime 单位是秒
     */
    public void set(String key, T value, long expireTime) {
    ValueOperations<String, T> valueOperations = redisTemplate.opsForValue();
    valueOperations.set(key,value,expireTime, TimeUnit.MINUTES);
    }
    
    /**
     * 判断 key 是否在 redis 数据库中
     *
     * @param key
     * @return
     */
    public boolean exists(final String key) {
    return redisTemplate.hasKey(key);
    }
    
    
    
    /**
     * 获取 key 对应的字符串
     * @param key
     * @return
     */
    public T get(String key) {
    ValueOperations<String, T> valueOperations = redisTemplate.opsForValue();
    return valueOperations.get(key);
    }
    
    /**
     * 获得 key 对应的键值，并更新缓存时间，时间长度为默认值
     * @param key
     * @return
     */
    public T getAndUpdateTime(String key) {
    T result = get(key);
    if (result != null) {
    set(key, result);
    }
    return result;
    }
    
    /**
     * 删除 key 对应的 value
     * @param key
     */
    public void delete(String key) {
    redisTemplate.delete(key);
    }
    
    
---
    
RedisConfig：

    import com.fasterxml.jackson.annotation.JsonAutoDetect;
    import com.fasterxml.jackson.annotation.PropertyAccessor;
    import com.fasterxml.jackson.databind.ObjectMapper;
    import org.springframework.context.annotation.Bean;
    import org.springframework.context.annotation.Configuration;
    import org.springframework.data.redis.connection.RedisConnectionFactory;
    import org.springframework.data.redis.core.RedisTemplate;
    import org.springframework.data.redis.core.StringRedisTemplate;
    import org.springframework.data.redis.serializer.Jackson2JsonRedisSerializer;
    
    @Configuration
    public class RedisConfig {
    
    @Bean
    public RedisTemplate<String, String> redisTemplate(RedisConnectionFactory redisConnectionFactory){
    StringRedisTemplate redisTemplate = new StringRedisTemplate(redisConnectionFactory);
    Jackson2JsonRedisSerializer jackson2JsonRedisSerializer = new Jackson2JsonRedisSerializer(Object.class);
    /**
     * 通用的序列化和反序列化设置
     * ObjectMapper类是Jackson库的主要类。它提供一些功能将转换成Java对象匹配JSON结构，反之亦然。
     */
    ObjectMapper objectMapper = new ObjectMapper();
    objectMapper.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.ANY);
    objectMapper.enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL);
    
    jackson2JsonRedisSerializer.setObjectMapper(objectMapper);
    redisTemplate.setValueSerializer(jackson2JsonRedisSerializer);
    redisTemplate.afterPropertiesSet();
    return redisTemplate;
    }
    
    

