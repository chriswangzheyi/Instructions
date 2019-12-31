package com.wzy.provider_demo2;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;

@EnableDiscoveryClient
@SpringBootApplication
public class ProviderDemo2Application {

    public static void main(String[] args) {
        SpringApplication.run(ProviderDemo2Application.class, args);
    }

}
