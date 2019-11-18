package com.wzy.springboot_redis_idempotence.controller;

import com.alibaba.fastjson.JSONObject;

import com.wzy.springboot_redis_idempotence.pojo.TokenInfo;
import com.wzy.springboot_redis_idempotence.pojo.User;
import com.wzy.springboot_redis_idempotence.service.TokenService;
import com.wzy.springboot_redis_idempotence.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.servlet.http.HttpServletRequest;

@Controller
@RequestMapping("/user")
public class UserController {

    @Autowired
    private UserService userService;

    @Autowired
    private TokenService tokenService;

    @RequestMapping("/login")
    @ResponseBody
    public String login(String username, String password, HttpServletRequest request) {
        TokenInfo dto = new TokenInfo();
        User user = this.userService.login(username, password);
        if (user != null) {
            String userAgent = request.getHeader("user-agent");
            String token = this.tokenService.generateToken(userAgent, username);
            this.tokenService.save(token, user);

            dto.setIsLogin("true");
            dto.setToken(token);
            dto.setTokenCreatedDate(System.currentTimeMillis());
            dto.setTokenExpiryDate(System.currentTimeMillis() + 2*60*60*1000);
        } else {
            dto.setIsLogin("false");
        }
        return JSONObject.toJSONString(dto);
    }
}
