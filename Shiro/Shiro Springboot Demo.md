#Shiro Springboot Demo


## 数据库设计

![](../Images/4.png)

	t_permission表：

![](../Images/5.png)

	t_role表：

![](../Images/6.png)

	t_role_permission表：

![](../Images/7.png)

	t_user表：

![](../Images/8.png)

	t_user_role表：

![](../Images/9.png)


## 代码

 见Demo/shiro_springboot_demo

## 访问目录

	localhost:9091

## 自定义过滤器

	#anon 游客模式，不登录也可以访问
	filterMap.put(???,"anon");

	#authc 标识只有登录用户才可以访问
	filterMap.put(???,"authc");

	#管理员角色才可以访问
	filterMap.put(???,"roleOrFilter[admin,root]");

	#按角色关键字
	filterMap.put(???,"perms[video_update]");


## 注意事项

	用户密码用md5加密，所以需要在测试类中，使用testUpdatePassword类更新密码