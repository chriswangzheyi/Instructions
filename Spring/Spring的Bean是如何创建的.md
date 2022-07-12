# Spring的Bean是如何创建的

1、根据Context类型（xml、注解）初始化BeanDefinitionReader,通过BeanDefinitionReader确认哪些Bean需要被初始化，然后将一个个的bean信息封装层BeanDefinition，最后在包装成BeanDefinitionWrapper，放入到BeanDefinitionRegister对应map中。

2、遍历所有BeanDefinition，按照 实例化-> 依赖注入->初始化->AOP的顺序通过反射去创建bean对象

3、将创建完成的Bean放入到一级缓存中存储，供用户使用。