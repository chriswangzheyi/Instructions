package com.wzy.springboot_druid.config;

import com.alibaba.druid.pool.DruidDataSource;
import com.alibaba.druid.support.http.StatViewServlet;
import com.alibaba.druid.support.http.WebStatFilter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.boot.web.servlet.ServletRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;
import java.util.HashMap;
import java.util.Map;

@Configuration
public class Configs {
    // 初始化druidDataSource对象
    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
    public DataSource druidDataSource(){
        return  new DruidDataSource();
    }


    // 注册后台监控界面
    @Bean
    public ServletRegistrationBean servletRegistrationBean(){
        // 绑定后台监控界面的路径  为localhost/druid
        ServletRegistrationBean bean=new ServletRegistrationBean(new StatViewServlet(),"/druid/*");
        Map<String,String>map=new HashMap<>();
        // 设置后台界面的用户名
        map.put("loginUsername","admin");
        //设置后台界面密码
        map.put("loginPassword","admin");
        // 设置那些ip允许访问，" " 为所有
        map.put("allow","");
        // 不允许该ip访问
        map.put("deny","33.32.43.123");
        bean.setInitParameters(map);
        return bean;
    }

    // 监听获取应用的数据，filter用于收集数据，servlet用于数据展示

    @Bean
    public FilterRegistrationBean filterRegistrationBean(){
        FilterRegistrationBean bean=new FilterRegistrationBean();
        // 设置过滤器
        bean.setFilter(new WebStatFilter());
        // 对所有请求进行过滤捕捉监听
        bean.addUrlPatterns("/*");
        Map<String,String> map=new HashMap<>();
        // 排除 . png  .js 的监听  用于排除一些静态资源访问
        map.put("exclusions","*.png，*.js");
        bean.setInitParameters(map);
        return bean;
    }

}
