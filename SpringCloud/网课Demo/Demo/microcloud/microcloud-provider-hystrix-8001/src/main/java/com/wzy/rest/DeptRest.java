package com.wzy.rest;

import com.netflix.hystrix.contrib.javanica.annotation.HystrixCommand;
import com.wzy.service.IDeptService;
import com.wzy.vo.Dept;
import org.springframework.web.bind.annotation.*;
import javax.annotation.Resource;


@RestController
public class DeptRest {
	@Resource
	private IDeptService deptService ;

	//正常的请求
	@RequestMapping(value="/dept/get/{id}",method=RequestMethod.GET)
	@HystrixCommand(fallbackMethod = "getForback")
	public Object get(@PathVariable("id") long id) {

		Dept vo = this.deptService.get(id) ;	// 接收数据库的查询结果
		if (vo == null) {	// 数据不存在，假设让它抛出个错误
			throw new RuntimeException("部门信息不存在！") ;
		}

		return this.deptService.get(id) ;
	}

	@RequestMapping(value="/dept/add",method=RequestMethod.GET)
	@HystrixCommand(fallbackMethod = "addForback")
	public Object add(@RequestBody Dept dept) {
		return this.deptService.add(dept) ;
	}

	@RequestMapping(value="/dept/list",method=RequestMethod.GET)
	@HystrixCommand(fallbackMethod = "listForback")
	public Object list() {
		return this.deptService.list() ;
	}


	//Fallback
	public Object getForback(@PathVariable("id") long id) {
		return new Dept() ;
	}


	public Object addForback(@RequestBody Dept dept) {
		return false ;
	}


	public Object listForback() {
		return null;
	}

}
