package com.wzy.service.impl;

import java.util.List;

import javax.annotation.Resource;

import com.wzy.dao.UserMapper;
import com.wzy.domain.User;
import com.wzy.service.UserService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;



@Service
@Transactional
public class UserServiceImpl implements UserService {

	@Resource
	private UserMapper userMapper;
	
	@Override
	public User findByName(String name) {
		return userMapper.findByName(name);
	}

	@Override
	public List<String> findPermissionByUserId(Integer userId) {
		return userMapper.findPermissionByUserId(userId);
	}

}
