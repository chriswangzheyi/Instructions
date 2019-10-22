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
