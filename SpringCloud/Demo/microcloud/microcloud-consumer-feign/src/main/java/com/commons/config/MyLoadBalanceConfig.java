package com.commons.config;

import org.springframework.context.annotation.Bean;

import com.netflix.loadbalancer.IRule;

public class MyLoadBalanceConfig {
	@Bean
	public IRule ribbonRule() { // 其中IRule就是所有规则的标准
		return new com.netflix.loadbalancer.RandomRule() ;
	}
}
