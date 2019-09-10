package com.wzy;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.circuitbreaker.EnableCircuitBreaker;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.cloud.netflix.eureka.EnableEurekaClient;
@SpringBootApplication
@EnableEurekaClient
@EnableCircuitBreaker
@EnableDiscoveryClient
public class Upload_8201_StartSpringCloudApplication {
	public static void main(String[] args) {
		SpringApplication.run(Upload_8201_StartSpringCloudApplication.class, args);
	}
}
