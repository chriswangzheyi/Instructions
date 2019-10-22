#Shiro Maven Demo

##项目结构

![](../Images/3.png)


## 代码

**MyRealm：**

	package com.wzy.realm;
	
	import org.apache.shiro.authc.AuthenticationException;
	import org.apache.shiro.authc.AuthenticationInfo;
	import org.apache.shiro.authc.AuthenticationToken;
	import org.apache.shiro.authc.SimpleAuthenticationInfo;
	import org.apache.shiro.authc.UsernamePasswordToken;
	import org.apache.shiro.authz.AuthorizationInfo;
	import org.apache.shiro.authz.SimpleAuthorizationInfo;
	import org.apache.shiro.realm.AuthorizingRealm;
	import org.apache.shiro.subject.PrincipalCollection;
	
	/**
	 * 自定义Realm
	 * @author lenovo
	 *
	 */
	public class MyRealm extends AuthorizingRealm{
	
	    //授权方法
	    @Override
	    protected AuthorizationInfo doGetAuthorizationInfo(PrincipalCollection arg0) {
	        System.out.println("ִ执行授权方法...");
	
	        SimpleAuthorizationInfo info = new SimpleAuthorizationInfo();
	
	        //资源的授权码
	        //info.addStringPermission("productAdd");
	
	        //通配符的授权
	        info.addStringPermission("product:*");
	
	        //角色的授权码
	        info.addRole("admin");
	
	        return info;
	    }
	
	    //认证方法
	    @Override
	    protected AuthenticationInfo doGetAuthenticationInfo(AuthenticationToken arg0) throws AuthenticationException {
	        System.out.println("执行认证方法...");
	
	        //判断用户名或密码
	
	        //1.获取用户输入的账户信息
	        UsernamePasswordToken token = (UsernamePasswordToken)arg0;
	        String username = token.getUsername();
	
	        //2.模拟数据库的账户信息
	        String name = "jack";
	        String password = "1234";
	
	
	        if(!username.equals(name)){
	            return null; // shiro底层自动抛出UnknownAccountException
	        }
	
	        return new SimpleAuthenticationInfo("callback",password,"");
	    }

	}



**Demo1：**


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


**Demo2：**

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
	 * 执行Shiro的授权流程
	 * @author lenovo
	 *
	 */
	public class Demo2 {
	
	    public static void main(String[] args) {
	        //1.创建安全管理器工厂
	        Factory<org.apache.shiro.mgt.SecurityManager> factory = new IniSecurityManagerFactory("classpath:shiro.ini");
	
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
	
	
	            //进行Shiro的授权
	            //1.基于资源的授权
	            //判断当前登录用户是否有“商品添加”功能
	            //isPermitted():返回true,有权限， false：没有权限
	            System.out.println("productAdd="+subject.isPermitted("product:add"));
	            System.out.println("productUpdate="+subject.isPermitted("product:update"));
	
	            //2.基于角色的授权
	            //判断当前登录用户是否为“超级管理员”
	            System.out.println("admin="+subject.hasRole("admin"));
	
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




**shiro.ini：**

	myRealm=com.wzy.realm.MyRealm
	
	securityManager.realm=$myRealm


**pom.xml：**

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0"
	         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	
	    <groupId>com.wzy</groupId>
	    <artifactId>Shiro-java</artifactId>
	    <version>1.0-SNAPSHOT</version>
	
	    <dependencies>
	        <!-- shiro相关 -->
	        <dependency>
	            <groupId>org.apache.shiro</groupId>
	            <artifactId>shiro-all</artifactId>
	            <version>1.3.2</version>
	        </dependency>
	        <dependency>
	            <groupId>commons-logging</groupId>
	            <artifactId>commons-logging</artifactId>
	            <version>1.2</version>
	        </dependency>
	    </dependencies>
	
	</project>