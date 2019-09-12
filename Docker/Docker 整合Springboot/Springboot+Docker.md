**Springboot 整合 Docker**



1**.新建一个Springboot 项目**


![](../Images/1.png)

---

**HelloController:**

	package com.wzy.docker_springboot_demo.controller;
	
	import org.springframework.web.bind.annotation.RequestMapping;
	import org.springframework.web.bind.annotation.RequestMethod;
	import org.springframework.web.bind.annotation.RestController;
	
	@RestController
	public class HelloController {
	
	    @RequestMapping(value = "/hi", method = RequestMethod.GET)
	    public String hello() {
	        return "hi";
	    }
	
	}



**DockerSpringbootDemoApplication:**

	package com.wzy.docker_springboot_demo;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class DockerSpringbootDemoApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(DockerSpringbootDemoApplication.class, args);
	    }
	
	}


**application.properties：**
    
    server.port=8080



打包后，运行jar包，

测试：

请求

    http://localhost:8080/hi



**2. 构建docker image**

前提条件： 需要具备docker环境的服务器


将打包好的jar上传至服务器，并重命名为app.jar(方便编写dockkerfile)

在同一个文件夹下编写Dockerfile
	
	FROM java:8
	MAINTAINER "zheyi"<123348687@qq.com>
	ADD app.jar app.jar
	EXPOSE 8080
	CMD java -jar app.jar




![](../Images/2.png)



此时文件夹下有app.jar以及 Dockerfile



然后构建Docker image:

    docker build -t zheyitest .



完毕后确认是否构建成功：

    docker images



.![](../Images/3.png)


启动docker

    docker run -d -p 8080:8080 zheyitest


**3.验证**

访问服务

{ip}:8080/hi

例如：

    http://47.112.142.231:8080/hi





