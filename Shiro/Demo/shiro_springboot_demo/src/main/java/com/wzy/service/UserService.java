package com.wzy.service;

import com.wzy.domain.User;

import java.util.List;



public interface UserService {
	public User findByName(String name);
	
	public List<String> findPermissionByUserId(Integer userId);
}
