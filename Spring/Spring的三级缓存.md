# Spring的三级缓存

## 循环依赖

循环依赖就是指循环引用，是两个或多个Bean相互之间的持有对方的引用。循环依赖有三种形态：

1、相互依赖，也就是A 依赖 B，B 又依赖 A，它们之间形成了循环依赖。

2、三者间依赖，也就是A 依赖 B，B 依赖 C，C 又依赖 A，形成了循环依赖。

3、自我依赖，也是A依赖A形成了循环依赖自己依赖自己。


## 缓存

###一级缓存（singletonObjects）

一级缓存用于存放已经实例化、初始化完成的Bean

###二级缓存（earlySingletonObjects）

二级缓存用于存放已经实例化,但未初始化的Bean.保证一个类多次循环依赖时仅构建一次保证单例

###三级缓存（singletonFactories）

三级缓存用于存放该Bean的BeanFactory,当加载一个Bean会先将该Bean包装为BeanFactory放入三级缓存

