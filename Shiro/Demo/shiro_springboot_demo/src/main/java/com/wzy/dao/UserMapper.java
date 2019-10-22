package com.wzy.dao;

import com.wzy.domain.User;

import java.util.List;



public interface UserMapper {

	/**
	 * 根据用户名查询用户
	 * @param name
	 * @return
	 */
	public User findByName(String name);
	
	/**
	 * 根据用户ID查询用户拥有的资源授权码
	 */
	public List<String> findPermissionByUserId(Integer userId);
	
	/**
	 * 更新用户密码的方法
	 */
	public void updatePassword(User user);
	
}
