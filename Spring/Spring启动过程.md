# Spring启动过程

* 读取web.xml文件。
* 创建 ServletContext，为 ioc 容器提供宿主环境。
* 触发容器初始化事件，调用 contextLoaderListener.contextInitialized()方法，在这个方法会初始化一个应用上下文WebApplicationContext，即 Spring 的 ioc 容器。ioc 容器初始化完成之后，会被存储到 ServletContext 中。
* 初始化web.xml中配置的Servlet。如DispatcherServlet，用于匹配、处理每个servlet请求。