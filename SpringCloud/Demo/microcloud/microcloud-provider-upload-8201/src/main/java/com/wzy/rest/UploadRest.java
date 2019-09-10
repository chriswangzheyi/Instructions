package com.wzy.rest;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.netflix.hystrix.contrib.javanica.annotation.HystrixCommand;

@RestController
public class UploadRest {
	@RequestMapping(value = "/upload", method = RequestMethod.POST)
	@HystrixCommand(fallbackMethod="uploadFallback")
	public String upload(@RequestParam("photo") MultipartFile photo) {
		if (photo != null) {	// 表示现在已经有文件上传了
			System.out.println("【*** UploadRest ***】文件名称："
					+ photo.getOriginalFilename() + "、文件大小：" + photo.getSize());
		}
		return "mldn-file-" + System.currentTimeMillis() + ".jpg" ;
	}
	public String uploadFallback(@RequestParam("photo") MultipartFile photo) {
		return "nophoto.jpg" ;
	}
}
