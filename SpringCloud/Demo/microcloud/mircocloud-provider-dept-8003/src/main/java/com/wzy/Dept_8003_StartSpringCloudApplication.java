package com.wzy;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.eureka.EnableEurekaClient;

@SpringBootApplication
@EnableEurekaClient
public class Dept_8003_StartSpringCloudApplication {
	public static void main(String[] args) {
		SpringApplication.run(Dept_8003_StartSpringCloudApplication.class, args);
	}
}