package com.wz.springboot_aop.Aspect;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.Pointcut;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import javax.servlet.http.HttpServletRequest;

@Aspect
@Component
public class DoHomeWorkAspect {
    /** 定义切入点 */
    @Pointcut("execution(* com.wz.springboot_aop.controller.DoHomeWorkController.doHomeWork(..))")
    public void homeWorkPointcut() {
    }

    /** 定义Before advice通知类型处理方法 */
    @Before("homeWorkPointcut()")
    public void beforeHomeWork() {
        ServletRequestAttributes requestAttributes = (ServletRequestAttributes) RequestContextHolder
                .getRequestAttributes();
        HttpServletRequest request = requestAttributes.getRequest();
        System.out.println(request.getParameter("name") + "想先吃个冰淇淋......");
    }

    /** 定义方法前后的处理方法 */
    @Around("homeWorkPointcut()")
    public void around(ProceedingJoinPoint joinPoint) {
        System.out.println("环绕通知，方法执行前");
        try {
            joinPoint.proceed();
        } catch (Throwable e) {
            e.printStackTrace();
        }
        System.out.println("环绕通知，方法执行后");
    }
}
