package com.wzy.provider_demo2.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController {

    @GetMapping("/test/{info}")
    public String test(@PathVariable("info") String info){
        return "provider2 returns" + info;
    }

}
