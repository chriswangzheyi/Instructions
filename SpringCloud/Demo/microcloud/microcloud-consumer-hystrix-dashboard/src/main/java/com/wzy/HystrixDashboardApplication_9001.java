package com.wzy;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.hystrix.dashboard.EnableHystrixDashboard;
@SpringBootApplication
@EnableHystrixDashboard
public class HystrixDashboardApplication_9001 {
	public static void main(String[] args) {
		SpringApplication.run(HystrixDashboardApplication_9001.class, args);
	}
}
