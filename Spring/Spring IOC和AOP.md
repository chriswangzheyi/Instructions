# Sping IOC 和 AOP

## 定义

IoC，英文全称Inversion of Control，意为控制反转。AOP，英文全称Aspect-Oriented Programming，意为面向切面编程。

## IOC

控制反转，简单点说，就是创建对象的控制权，被反转到了Spring框架上。

通常，我们实例化一个对象时，都是使用类的构造方法来new一个对象，这个过程是由我们自己来控制的，而控制反转就把new对象的工交给了Spring容器。

Spring的IOC有三种注入方式 ：构造器注入、setter方法注入、根据注解注入。


## AOP

用于将那些与业务无关，但却对多个对象产生影响的公共行为和逻辑，抽取并封装为一个可重用的模块，这个模块被命名为“切面”（Aspect），减少系统中的重复代码，降低了模块间的耦合度，同时提高了系统的可维护性。可用于权限认证、日志、事务处理。

###切面（Aspect）：
共有功能的实现。如日志切面、权限切面、验签切面等。在实际开发中通常是一个存放共有功能实现的标准Java类。当Java类使用了@Aspect注解修饰时，就能被AOP容器识别为切面。

### 通知（Advice）：
切面的具体实现。就是要给目标对象织入的事情。以目标方法为参照点，根据放置的地方不同，可分为前置通知（Before）、后置通知（AfterReturning）、异常通知（AfterThrowing）、最终通知（After）与环绕通知（Around）5种。在实际开发中通常是切面类中的一个方法，具体属于哪类通知，通过方法上的注解区分。

###连接点（JoinPoint）：
程序在运行过程中能够插入切面的地点。例如，方法调用、异常抛出等。Spring只支持方法级的连接点。一个类的所有方法前、后、抛出异常时等都是连接点。

### 切入点（Pointcut）：
用于定义通知应该切入到哪些连接点上。不同的通知通常需要切入到不同的连接点上，这种精准的匹配是由切入点的正则表达式来定义的。

比如，在上面所说的连接点的基础上，来定义切入点。我们有一个类，类里有10个方法，那就产生了几十个连接点。但是我们并不想在所有方法上都织入通知，我们只想让其中的几个方法，在调用之前检验下入参是否合法，那么就用切点来定义这几个方法，让切点来筛选连接点，选中我们想要的方法。切入点就是来定义哪些类里面的哪些方法会得到通知。

### 目标对象（Target）：
那些即将切入切面的对象，也就是那些被通知的对象。这些对象专注业务本身的逻辑，所有的共有功能等待AOP容器的切入。

###代理对象（Proxy）：
将通知应用到目标对象之后被动态创建的对象。可以简单地理解为，代理对象的功能等于目标对象本身业务逻辑加上共有功能。代理对象对于使用者而言是透明的，是程序运行过程中的产物。目标对象被织入共有功能后产生的对象。

###织入（Weaving）：
将切面应用到目标对象从而创建一个新的代理对象的过程。这个过程可以发生在编译时、类加载时、运行时。Spring是在运行时完成。


#### 例子

	@Aspect
	@Component
	public class SignAop {
	 
	    @Pointcut("execution(public * cn.wbnull.springbootdemo.controller.*.*(..))")
	    private void signAop() {
	 
	    }
	 
	    @Before("signAop()")
	    public void doBefore(JoinPoint joinPoint) throws Exception {
	        //code
	       }
	 
	    @AfterReturning(value = "signAop()", returning = "params")
	    public JSONObject doAfterReturning(JoinPoint joinPoint, JSONObject params) {
	        //code
	        }
	}