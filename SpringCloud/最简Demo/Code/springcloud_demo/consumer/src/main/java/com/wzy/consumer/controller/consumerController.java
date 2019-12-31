package com.wzy.consumer.controller;

import com.wzy.consumer.service.consumerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class consumerController {

    @Autowired
    consumerService consumerService;

    @RequestMapping("/consumer")
    public String testFeign(){
        return consumerService.testFeign("1111");
    }

}
