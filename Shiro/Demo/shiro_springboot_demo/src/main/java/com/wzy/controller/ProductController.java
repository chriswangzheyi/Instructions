package com.wzy.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/product")
public class ProductController {

	@RequestMapping("/toAdd")
	public String toAdd(){
		return "product/add";
	}
	
	@RequestMapping("/toList")
	public String toList(){
		return "product/list";
	}
	
	@RequestMapping("/toUpdate")
	public String toUpdate(){
		return "product/update";
	}
}
