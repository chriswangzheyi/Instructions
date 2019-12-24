package com.wz.springboot_aop.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class DoHomeWorkController {
    @GetMapping("/dohomework")
    public void doHomeWork(String name) {
        System.out.println(name + "做作业... ...");
    }
}
