package com.wzy.config;

import com.wzy.filter.AuthorizedRequestFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;



@Configuration
public class ZuulConfig {
	@Bean
	public AuthorizedRequestFilter getAuthorizedRequestFilter() {
		return new AuthorizedRequestFilter() ;
	}
}
