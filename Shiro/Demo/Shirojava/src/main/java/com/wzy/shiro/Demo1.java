package com.wzy.shiro;

import org.apache.shiro.SecurityUtils;
import org.apache.shiro.authc.AuthenticationToken;
import org.apache.shiro.authc.IncorrectCredentialsException;
import org.apache.shiro.authc.UnknownAccountException;
import org.apache.shiro.authc.UsernamePasswordToken;
import org.apache.shiro.config.IniSecurityManagerFactory;
import org.apache.shiro.mgt.SecurityManager;
import org.apache.shiro.subject.Subject;
import org.apache.shiro.util.Factory;

/**
 * 执行Shiro的认证流程
 * @author lenovo
 *
 */
public class Demo1 {

	public static void main(String[] args) {
		//1.创建安全管理器工厂
		Factory<SecurityManager> factory = new IniSecurityManagerFactory("classpath:shiro.ini");
		
		//2.创建安全管理器
		SecurityManager securityManager = factory.getInstance();
		
		//3.初始化SecurityUtils工具类
		SecurityUtils.setSecurityManager(securityManager);
		
		//4.从SecurityUtils工具中获取Subject
		Subject subject = SecurityUtils.getSubject();
		
		//5.认证操作（登录）
		//AuthenticationToken: 用于封装用户输入的账户信息
		AuthenticationToken token = new UsernamePasswordToken("jack", "1234");
		
		try {
			subject.login(token);
			
			//获取SimpleAuthenticationInfo的第一个参数：principal
			Object principal = subject.getPrincipal();
			
			//如果login方法没有任何异常，代表认证成功
			System.out.println("登录成功："+principal);
		} catch (UnknownAccountException e) {
			//账户不存在
			System.out.println("账户不存在");
		}  catch (IncorrectCredentialsException e) {
			//密码错误
			System.out.println("密码错误");
		} catch (Exception e) {
			//系统错误
			System.out.println("系统错误");
		} 
	}
}
