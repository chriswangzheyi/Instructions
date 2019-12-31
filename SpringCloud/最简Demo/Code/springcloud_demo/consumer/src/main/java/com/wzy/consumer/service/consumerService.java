package com.wzy.consumer.service;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

@FeignClient(name = "PROVIDER") //需要跟provider在eureka中注册的名字一样
public interface consumerService {

    @RequestMapping("/test/{info}")  //需要跟provider的Url保持一致
    public String testFeign(@PathVariable("info") String info);

}
