package com.wzy.springboot_aop2.Aspect;

import org.aspectj.lang.annotation.After;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.stereotype.Component;

@Component
@Aspect
public class TestAspect {

    @Pointcut("execution(public * com.wzy.springboot_aop2.controller.*.*(..))")
    public void TestPointCut(){
    }

    @Before("TestPointCut()")
    public void beforeTest(){
        System.out.println("进入切片之前");
    }

    @After("TestPointCut()")
    public void afterTest(){
        System.out.println("进入切片之后");
    }
}
