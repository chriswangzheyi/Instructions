package com.wzy.provider_demo2.controller;

import com.netflix.hystrix.contrib.javanica.annotation.HystrixCommand;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController {

    @GetMapping("/test/{info}")
    @HystrixCommand(fallbackMethod = "getFallBack")
    public String test(@PathVariable("info") String info){
        return "provider2 returns" + info;
    }

    //fallback方法的请求报文和返回结构需要跟原方法一致
    public String getFallBack(String info){
        return "fallback "+ info;
    }
}
