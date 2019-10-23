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

## 注意事项

	用户密码用md5加密，所以需要在测试类中，使用testUpdatePassword类更新密码