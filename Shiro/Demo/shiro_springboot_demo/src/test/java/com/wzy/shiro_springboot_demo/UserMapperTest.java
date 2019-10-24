package com.wzy.shiro_springboot_demo;

import javax.annotation.Resource;

import com.wzy.ShiroSpringbootDemoApplication;
import com.wzy.dao.UserMapper;
import com.wzy.domain.User;
import org.apache.shiro.crypto.hash.Md5Hash;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;



@RunWith(SpringJUnit4ClassRunner.class)
@SpringBootTest(classes= ShiroSpringbootDemoApplication.class)
public class UserMapperTest {

	@Resource
	private UserMapper userMapper;
	
	@Test
	public void testFindByName(){
		User user = userMapper.findByName("eric");
		System.out.println(user);
	}
	
	@Test
	public void testUpdatePassword(){
		User user = new User();
		user.setId(1);
		
		//使用shiro的加密工具类进行密码加密
		Md5Hash hash = new Md5Hash("123456", "eric", 2);
		
		user.setPassword(hash.toString());
		user.setName("eric");
		userMapper.updatePassword(user);
	}
}
