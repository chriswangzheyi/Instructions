package com.wzy;

import java.util.List;

import com.wzy.fallback.IDeptClientServiceFallback;
import com.wzy.vo.Dept;
import org.springframework.cloud.netflix.feign.FeignClient;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import com.commons.config.FeignClientConfig;

@FeignClient(value="MICROCLOUD-PROVIDER-DEPT",configuration=FeignClientConfig.class, fallbackFactory = IDeptClientServiceFallback.class)
public interface IDeptClientService {

	@RequestMapping(method=RequestMethod.GET,value="/dept/get/{id}")
	public Dept get(@PathVariable("id") long id) ;

	@RequestMapping(method=RequestMethod.GET,value="/dept/list")
	public List<Dept> list() ;

	@RequestMapping(method=RequestMethod.POST,value="/dept/add")
	public boolean add(Dept dept) ;
}
