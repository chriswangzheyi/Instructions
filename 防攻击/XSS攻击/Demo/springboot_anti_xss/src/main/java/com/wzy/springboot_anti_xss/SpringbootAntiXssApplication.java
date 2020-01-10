package com.wzy.springboot_anti_xss;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.ServletComponentScan;

@SpringBootApplication
@ServletComponentScan
public class SpringbootAntiXssApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringbootAntiXssApplication.class, args);
    }

}
