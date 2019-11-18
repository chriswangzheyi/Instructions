package com.wzy.springboot_redis_idempotence.service;

import com.wzy.springboot_redis_idempotence.pojo.User;
import org.springframework.stereotype.Service;

@Service("userService")
public class UserService {
    public User login(String username, String password) {
        if ("tom".equals(username) && "123".equals(password)){
            return new User(1, "tom", "123");
        } else {
            return null;
        }
    }
}

