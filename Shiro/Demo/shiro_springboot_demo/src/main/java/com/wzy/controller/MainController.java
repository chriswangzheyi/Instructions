package com.wzy.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

/**
 * 主控制器
 * @author lenovo
 *
 */
@Controller
@RequestMapping("/")
public class MainController {

	@RequestMapping("/index")
	public String index(){
		return "index";
	}
	
	@RequestMapping("/toLogin")
	public String toLogin(){
		return "login";
	}
	
	@RequestMapping("/unAuth")
	public String unAuth(){
		return "unauth";
	}
}
