package com.wzy.config;

import javax.annotation.Resource;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.builders.WebSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.config.http.SessionCreationPolicy;
@Configuration
@EnableWebSecurity
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {
	@Resource
	public void configGlobal(AuthenticationManagerBuilder auth)
			throws Exception {
		auth.inMemoryAuthentication().withUser("mldnjava").password("hello")
				.roles("USER").and().withUser("admin").password("hello")
				.roles("USER", "ADMIN");
	}
	@Override
	protected void configure(HttpSecurity http) throws Exception {
		// 表示所有的访问都必须进行认证处理后才可以正常进行
		http.httpBasic().and().authorizeRequests().anyRequest()
				.fullyAuthenticated().and().csrf().disable();
		// 所有的Rest服务一定要设置为无状态，以提升操作性能
		http.sessionManagement()
				.sessionCreationPolicy(SessionCreationPolicy.STATELESS);
	}

	@Override
	public void configure(WebSecurity web) throws Exception {
		web.ignoring().antMatchers("/hystrix.stream","/turbine.stream") ;
	}

}
