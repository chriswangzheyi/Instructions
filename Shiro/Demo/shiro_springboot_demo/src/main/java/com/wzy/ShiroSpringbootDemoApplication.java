package com.wzy;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.wzy.dao")
public class ShiroSpringbootDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(ShiroSpringbootDemoApplication.class, args);
    }

}
