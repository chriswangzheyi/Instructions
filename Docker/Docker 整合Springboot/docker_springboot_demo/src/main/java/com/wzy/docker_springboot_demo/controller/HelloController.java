package com.wzy.docker_springboot_demo.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HelloController {

    @RequestMapping(value = "/hi", method = RequestMethod.GET)
    public String hello() {
        return "hi";
    }

}
