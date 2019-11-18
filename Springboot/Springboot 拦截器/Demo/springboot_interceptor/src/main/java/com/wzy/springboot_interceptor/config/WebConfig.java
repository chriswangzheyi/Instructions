package com.wzy.springboot_interceptor.config;

import com.wzy.springboot_interceptor.interceptor.LoginInterceptor;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurationSupport;

@Configuration
public class WebConfig extends WebMvcConfigurationSupport {


    @Override
    protected void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(new LoginInterceptor())
                .addPathPatterns("/**")
                .excludePathPatterns("/login");
        //这里的配置意思是会拦截所有的请求，除了login的。拦截的请求会根据LoginInterceptor所写逻辑处理
    }
}
