# XSS攻击

---

## 原理

XSS 全称 Cross Site Scripting ，跨站脚本攻击

![](../Images/1.png)


## 主要危害

　　1、盗取各类用户帐号，如机器登录帐号、用户网银帐号、各类管理员帐号

　　2、控制企业数据，包括读取、篡改、添加、删除企业敏感数据的能力

　　3、盗窃企业重要的具有商业价值的资料

　　4、非法转账

　　5、强制发送电子邮件

　　6、网站挂马

　　7、控制受害者机器向其它网站发起攻击

## 攻击方式

### 反射型：经过后端，不经过数据库

也叫非持久型XSS，交互数据一般不会被存在数据库里面，一次性，所见即所得。一般XSS代码出现在请求URL中，作为参数提交到服务器，服务器解析并响应，响应结果中包含XSS代码，最后浏览器解析并执行。

场景：

1、用户A给用户B发送一个恶意构造了Web的URL。

2、用户B点击并查看了这个URL。

3、用户B获取到一个具有漏洞的HTML页面并显示在本地浏览器中。

4、漏洞HTML页面执行恶意JavaScript脚本，将用户B信息盗取发送给用户A，或者篡改用户B看到的数据等。


### 存储型：经过后端，经过数据库



### DOM：不经过后端,DOM- xss是通过url传入参数去控制触发的。