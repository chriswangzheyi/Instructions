package com.wzy.springboot_aop2.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController2 {

    @GetMapping("/test2")
    public String test2(){
        System.out.println("22222");
        return "22";
    }
}
