**Mongodb整合Sprongboot**



**1.项目结构**

通过mongoTemplate进行CRUD


![](../Images/1.png)



**MongodbController:**

	import com.wzy.mongodbdemo.vo.User;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.data.mongodb.core.MongoTemplate;
	import org.springframework.data.mongodb.core.query.Criteria;
	import org.springframework.data.mongodb.core.query.Query;
	import org.springframework.data.mongodb.core.query.Update;
	import org.springframework.web.bind.annotation.*;
	import java.util.List;
	
	
	@RestController
	@RequestMapping("/mongodbTest")
	public class MongodbController {
	
	    @Autowired
	    private MongoTemplate mongoTemplate;
	
	    /*表名*/
	    private static final String collectionName="user";
	
	    /**
	     * 描述：新增
	     * @author 汪哲逸
	     * @created 2018/9/1 20:17
	     * @param user
	     * @return Object
	     */
	    @RequestMapping(value = "/insert", method = RequestMethod.POST)
	    @ResponseBody
	    public Object insert(@ModelAttribute User user) throws Exception {
	      return   this.mongoTemplate.insert(user);
	    }
	
	    /**
	     * 描述：删除
	     * @author 汪哲逸
	     * @created 2018/9/1 20:17
	     * @param userId
	     * @return Object
	     */
	    @RequestMapping("/delete")
	    @ResponseBody
	    public Object delete(@RequestParam("userId") String userId) throws Exception {
	        Query query = Query.query(Criteria.where("userId").is(userId));
	        return this.mongoTemplate.remove(query, collectionName);
	    }
	
	    /**
	     * 描述：修改
	     * @author 汪哲逸
	     * @created 2018/9/1 20:17
	     * @param user
	     * @return Object
	     */
	    @RequestMapping(value = "/update", method = RequestMethod.POST)
	    @ResponseBody
	    public Object update(@ModelAttribute User user) throws Exception {
	        Query query = Query.query(Criteria.where("userId").is(user.getUserId()));
	        Update update = new Update();
	        update.set("age", user.getAge());
	        update.set("name", user.getName());
	        update.set("email", user.getEmail());
	        return this.mongoTemplate.updateFirst(query, update, collectionName);
	    }
	
	    /**
	     * 描述：查询
	     * @author 汪哲逸
	     * @created 2018/9/1 20:17
	     * @param
	     * @return Object
	     */
	    @RequestMapping("/query")
	    @ResponseBody
	    public Object query(@ModelAttribute User user) throws Exception {
	        Query query = Query.query(Criteria.where("userId").is(user.getUserId()));
	        List<User> users = this.mongoTemplate.find(query, User.class);
	        return users;
	    }
	
	
	}



**User:**

	
	import org.springframework.data.mongodb.core.mapping.Document;
	
	import java.util.Date;
	
	@Document(collection="user")
	public class User {
	
	    private String userId;
	
	    private String name;
	
	    private String uclass;
	
	    private String email;
	
	    private Date birthday;
	
	    private int age;
	
	    public String getUserId() {
	        return userId;
	    }
	
	    public void setUserId(String userId) {
	        this.userId = userId;
	    }
	
	    public String getName() {
	        return name;
	    }
	
	    public void setName(String name) {
	        this.name = name;
	    }
	
	    public String getUclass() {
	        return uclass;
	    }
	
	    public void setUclass(String uclass) {
	        this.uclass = uclass;
	    }
	
	    public String getEmail() {
	        return email;
	    }
	
	    public void setEmail(String email) {
	        this.email = email;
	    }
	
	    public Date getBirthday() {
	        return birthday;
	    }
	
	    public void setBirthday(Date birthday) {
	        this.birthday = birthday;
	    }
	
	    public int getAge() {
	        return age;
	    }
	
	    public void setAge(int age) {
	        this.age = age;
	    }
	}


**注意：@Document(collection="user") 直接对应mongodb的document名。**

---

**MongodbdemoApplication：**

	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class MongodbdemoApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(MongodbdemoApplication.class, args);
	    }
	
	}



**application.yml**


	spring:
	  data:
	    mongodb:
	      host: 47.112.142.231
	      port: 27017
	      database: test
	      username: root
	      password: root
	
	server:
	  port: 8082




**pom.xml**


	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.1.7.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>mongodbdemo</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>mongodbdemo</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-data-mongodb</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-web</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-test</artifactId>
	            <scope>test</scope>
	        </dependency>
	        <dependency>
	            <groupId>org.mybatis.spring.boot</groupId>
	            <artifactId>mybatis-spring-boot-starter</artifactId>
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

**3.设置mongdb的账号密码**


    mongo  
    use test   #test是目标database   
    db.createUser({user:"root",pwd:"root",roles:[{role:'root',db:'admin'}]})   #账号root ,密码root


**4.测试**

新增：

    http://localhost:8082/mongodbTest/insert?userId=015&name=Back&uclass=B&email=b12@sina.com&age=11&dataStatus=1


修改：

	http://localhost:8082/mongodbTest/update?userId=015&name=Back&uclass=B&email=b12@sina.com&age=18&dataStatus=2


查询：

	http://localhost:8082/mongodbTest/query?userId=015

删除

	http://localhost:8082/mongodbTest/delete?userId=015

