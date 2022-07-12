# Spring常用的设计模式有哪些

## 模板方法设计模式

参考：https://baijiahao.baidu.com/s?id=1736128832181642837&wfr=spider&for=pc

Spring中jdbcTemplate、hibernateTemplate等以Template结尾的对数据库操作的 类，它们就使用到模板模式。

## 单例设计模式

Spring中bean的默认作用域就是singleton。spring的一级缓存就是使用的容器式单例

## 代理设计模式

Spring AOP就是基于动态代理的。如果要代理的对象，实现了某个接口，那么Spring AOP 会使用JDK Proxy，去创建代理对象，而对于没有实现接口的对象，就无法使用JDK Proxy去进行代理了，这 时候Spring AOP会使用Cglib，这时候Spring AOP会使用Cglib生成一个被代理对象的子类来作为代理。

## 工厂设计模式

Spring使用工厂模式可以通过BeanFactory或ApplicationContext创建bean对象。

（未完待续）



