package com.wzy;

import org.springframework.boot.SpringApplication;
		import org.springframework.boot.autoconfigure.SpringBootApplication;
		import org.springframework.cloud.netflix.eureka.server.EnableEurekaServer;
@SpringBootApplication
@EnableEurekaServer
public class Eureka_7001_StartSpringCloudApplication {
	public static void main(String[] args) {
		SpringApplication.run(Eureka_7001_StartSpringCloudApplication.class, args);
	}
}
