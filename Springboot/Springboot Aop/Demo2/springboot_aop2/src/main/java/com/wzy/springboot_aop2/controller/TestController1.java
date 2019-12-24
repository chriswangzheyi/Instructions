package com.wzy.springboot_aop2.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController1 {

    @GetMapping("/test1")
    public String test1(){
        System.out.println("111111");
        return "11";
    }
}
