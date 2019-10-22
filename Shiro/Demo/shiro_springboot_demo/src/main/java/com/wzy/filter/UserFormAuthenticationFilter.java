package com.wzy.filter;

import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;

import com.wzy.domain.User;
import org.apache.shiro.session.Session;
import org.apache.shiro.subject.Subject;
import org.apache.shiro.web.filter.authc.FormAuthenticationFilter;


/**
 * 自定义认证过滤器，加入RememberMe的功能
 * @author lenovo
 *
 */
public class UserFormAuthenticationFilter extends FormAuthenticationFilter {

	@Override
	protected boolean isAccessAllowed(ServletRequest request, ServletResponse response, Object mappedValue) {
		Subject subject = getSubject(request, response);

		// 如果 isAuthenticated 为 false 证明不是登录过的，同时 isRememberd 为true
		// 证明是没登陆直接通过记住我功能进来的
		if (!subject.isAuthenticated() && subject.isRemembered()) {

			// 获取session看看是不是空的
			Session session = subject.getSession(true);

			// 查看session属性当前是否是空的
			if (session.getAttribute("userName") == null) {
				// 如果是空的才初始化
				User dbUser = (User)subject.getPrincipal();
				//存入用户数据
				session.setAttribute("userName", dbUser.getName());
			}
		}

		// 这个方法本来只返回 subject.isAuthenticated() 现在我们加上 subject.isRemembered()
		// 让它同时也兼容remember这种情况
		return subject.isAuthenticated() || subject.isRemembered();
	}
}
