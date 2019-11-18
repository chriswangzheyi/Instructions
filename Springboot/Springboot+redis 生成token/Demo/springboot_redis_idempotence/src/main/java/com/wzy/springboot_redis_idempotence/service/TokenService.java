package com.wzy.springboot_redis_idempotence.service;

import com.alibaba.fastjson.JSONObject;

import com.wzy.springboot_redis_idempotence.pojo.User;
import com.wzy.springboot_redis_idempotence.utils.RedisUtil;
import nl.bitwalker.useragentutils.UserAgent;
import org.apache.commons.codec.digest.DigestUtils;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

@Service("tokenService")
public class TokenService {

    @Resource
    private RedisUtil redisUtil;

    //生成token(格式为token:设备-加密的用户名)
    public String generateToken(String userAgentStr, String username) {
        StringBuilder token = new StringBuilder("token:");
        //设备
        UserAgent userAgent = UserAgent.parseUserAgentString(userAgentStr);
        if (userAgent.getOperatingSystem().isMobileDevice()) {
            token.append("MOBILE-");
        } else {
            token.append("PC-");
        }
        //加密的用户名作为redis的key
        token.append(DigestUtils.md5Hex(username) + "-");
        System.out.println("token-->" + token.toString());
        return token.toString();
    }

    //把token存到redis中
    public void save(String token, User user) {

        if (token.startsWith("token:PC")) {
            redisUtil.setex(token, JSONObject.toJSONString(user), 2*60*60);
        } else {
            redisUtil.set(token, JSONObject.toJSONString(user));
        }
    }

}
