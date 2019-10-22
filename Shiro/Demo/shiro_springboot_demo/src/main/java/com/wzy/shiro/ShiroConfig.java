package com.wzy.shiro;

import java.util.LinkedHashMap;
import java.util.Map;

import javax.servlet.Filter;

import com.wzy.filter.UserFormAuthenticationFilter;
import org.apache.shiro.spring.web.ShiroFilterFactoryBean;
import org.apache.shiro.web.mgt.CookieRememberMeManager;
import org.apache.shiro.web.mgt.DefaultWebSecurityManager;
import org.apache.shiro.web.servlet.SimpleCookie;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import at.pollux.thymeleaf.shiro.dialect.ShiroDialect;


/**
 * Shiro的配置类
 * @author lenovo
 *
 */
@Configuration
public class ShiroConfig {

	/**
	 * 1.创建ShiroFilterFactoryBean
	 */
	@Bean
	public ShiroFilterFactoryBean shiroFilterFactoryBean(DefaultWebSecurityManager securityManager){
		ShiroFilterFactoryBean bean = new ShiroFilterFactoryBean();
		
		//关联SecurityManager
		bean.setSecurityManager(securityManager);
		
		Map<String,String> filterMap = new LinkedHashMap<>();
		
		//认证过滤器
		filterMap.put("/product/toAdd", "perms[product:add]");
		//放行登录页面
		filterMap.put("/user/login", "anon");
		//验证码请求
		filterMap.put("/defaultKaptcha", "anon");
		//授权过滤器
		filterMap.put("/product/toList", "perms[product:list]");
		filterMap.put("/product/toUpdate", "perms[product:update]");
		//添加user过滤器
		filterMap.put("/index", "userFilter"); // /index的请求只要使用rememberMe功能，就可以访问了
		filterMap.put("/**", "authc");
		
		//把自定义Filter添加到shiro过滤器列表
		Map<String,Filter> filters = new LinkedHashMap<>();
		filters.put("userFilter", userFormAuthenticationFilter());
		bean.setFilters(filters);
		
		//添加shiro过滤器
		bean.setFilterChainDefinitionMap(filterMap);
		
		//修改登录请求
		bean.setLoginUrl("/toLogin");
		//添加未授权提示页面
		bean.setUnauthorizedUrl("/unAuth");
		return bean;
	}
	
	//创建自定义的过滤器
	@Bean(name="userFilter")
	public UserFormAuthenticationFilter userFormAuthenticationFilter(){
		return new UserFormAuthenticationFilter();
	}
	
	/**
	 * 2.创建SecurityManager
	 */
	@Bean
	public DefaultWebSecurityManager defaultWebSecurityManager(MyRealm realm,CookieRememberMeManager rememberMeManager){
		DefaultWebSecurityManager manager = new DefaultWebSecurityManager();
		
		//关联realm
		manager.setRealm(realm);
		
		//关联rememberMeManager
		manager.setRememberMeManager(rememberMeManager);
		
		return manager;
	}
	
	//创建CookieRememberMeManager
	@Bean
	public CookieRememberMeManager cookieRememberMeManager(SimpleCookie cookie){
		CookieRememberMeManager manager = new CookieRememberMeManager();
		manager.setCookie(cookie);
		return manager;
	}
	
	/**
	 * RememberMe的功能
	 */
	//创建Cookie
	@Bean
	public SimpleCookie simpleCookie(){
		SimpleCookie cookie = new SimpleCookie("rememberMe");
		//设置cookie的时间长度
		cookie.setMaxAge(120);
		//设置只读模型
		cookie.setHttpOnly(true);
		return cookie;
	}
	
	/**
	 * 3.创建Realm
	 */
	@Bean
	public MyRealm myReal(){
		MyRealm realm = new MyRealm();
		return realm;
	}
	
	/**
	 * 整合Shiro标签
	 */
	@Bean
	public ShiroDialect shiroDialect(){
		return new ShiroDialect();
	}
	
}
