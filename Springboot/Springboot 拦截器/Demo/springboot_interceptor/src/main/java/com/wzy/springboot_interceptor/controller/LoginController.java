package com.wzy.springboot_interceptor.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class LoginController {

        @GetMapping(value = "/login")
        public String login(){
            return "login request not intercept";
        }

}


