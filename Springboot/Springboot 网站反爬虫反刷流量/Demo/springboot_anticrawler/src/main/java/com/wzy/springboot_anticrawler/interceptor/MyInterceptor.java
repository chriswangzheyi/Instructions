package com.wzy.springboot_anticrawler.interceptor;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.concurrent.TimeUnit;
import org.apache.commons.codec.digest.DigestUtils;

@Component
public class MyInterceptor implements HandlerInterceptor {

    @Autowired
    private RedisTemplate redisTemplate;

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        String clientIP = request.getRemoteAddr();
        String userAgent = request.getHeader("User-Agent");
        String key = "anti:refresh:" + DigestUtils.md5Hex(clientIP + "_" + userAgent);
        response.setContentType("text/html;charset=utf-8");

        if(redisTemplate.hasKey("anti:refresh:blacklist")){
            if (redisTemplate.opsForSet().isMember("anti:refresh:blacklist", clientIP)) {
                response.getWriter().println("检测到您的IP访问异常，已被加入黑名单");
                System.out.println("检测到您的IP访问异常，已被加入黑名单");
                return false;
            }
        }

        //计数器
        Object keyNum =redisTemplate.opsForValue().get(key) ;
        Integer num = null;
        if(keyNum != null){
            num = Integer.valueOf(String.valueOf(keyNum));
        }

        if(num == null){ //第一次访问
            redisTemplate.opsForValue().set(key, String.valueOf(1), 60, TimeUnit.SECONDS);
        }else{

            if(num > 30 && num  < 100){
                response.getWriter().println("请求过于频繁，请稍后再试!");
                System.out.println("请求过于频繁，请稍后再试!");
                redisTemplate.opsForValue().increment(key, 1);
                return false;
            }else if(num >=100){
                response.getWriter().println("检测到您的IP访问异常，已被加入黑名单");
                System.out.println("检测到您的IP访问异常，已被加入黑名单");
                redisTemplate.opsForSet().add("anti:refresh:blacklist" , clientIP);
                return false;
            }else{
                redisTemplate.opsForValue().increment(key, 1);
            }
        }
        return true;
    }
}
