package com.wzy.shiro;

import java.util.List;

import javax.annotation.Resource;

import com.wzy.domain.User;
import com.wzy.service.UserService;
import org.apache.shiro.SecurityUtils;
import org.apache.shiro.authc.AuthenticationException;
import org.apache.shiro.authc.AuthenticationInfo;
import org.apache.shiro.authc.AuthenticationToken;
import org.apache.shiro.authc.SimpleAuthenticationInfo;
import org.apache.shiro.authc.UsernamePasswordToken;
import org.apache.shiro.authz.AuthorizationInfo;
import org.apache.shiro.authz.SimpleAuthorizationInfo;
import org.apache.shiro.realm.AuthorizingRealm;
import org.apache.shiro.subject.PrincipalCollection;
import org.apache.shiro.subject.Subject;
import org.springframework.util.StringUtils;



public class MyRealm extends AuthorizingRealm {

	@Resource
	private UserService userService;
	
	@Override
	protected AuthorizationInfo doGetAuthorizationInfo(PrincipalCollection principals) {
		SimpleAuthorizationInfo info = new SimpleAuthorizationInfo();

		//得到当前用户
		Subject subject = SecurityUtils.getSubject();
		User dbUser = (User)subject.getPrincipal();
		
		List<String> perms = userService.findPermissionByUserId(dbUser.getId());
		if(perms!=null){
			for (String perm : perms) {
				if(!StringUtils.isEmpty(perm)){
					info.addStringPermission(perm);
				}
			}
		}
		return info;
	}

	//认证
	@Override
	protected AuthenticationInfo doGetAuthenticationInfo(AuthenticationToken arg0) throws AuthenticationException {
		// 1.获取用户输入的账户信息
		UsernamePasswordToken token = (UsernamePasswordToken) arg0;

		User dbUser = userService.findByName(token.getUsername());

		if(dbUser==null){
			//用户不存在
			return null;
		}
		
		// 返回密码
		return new SimpleAuthenticationInfo(dbUser, dbUser.getPassword(), "");
	}

}
