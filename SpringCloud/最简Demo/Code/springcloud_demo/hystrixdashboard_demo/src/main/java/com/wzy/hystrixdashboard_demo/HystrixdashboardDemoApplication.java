package com.wzy.hystrixdashboard_demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.hystrix.dashboard.EnableHystrixDashboard;

// dashboard 观察用Strem例子如下： http://127.0.0.1:8001/hystrix.stream

@SpringBootApplication
@EnableHystrixDashboard
public class HystrixdashboardDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(HystrixdashboardDemoApplication.class, args);
    }

}
