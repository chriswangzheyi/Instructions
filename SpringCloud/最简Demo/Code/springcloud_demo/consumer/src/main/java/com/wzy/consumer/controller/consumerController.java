package com.wzy.consumer.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class consumerController {

    @Autowired
    com.wzy.consumer.service.consumerService consumerService;

    @RequestMapping("/consumer")
    public String testFeign(){
        return consumerService.testFeign("1111");
    }

}
