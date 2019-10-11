#Springboot 多线程 Demo


## 项目结构

![](../Images/1.png)


## 文件说明

**AsyncConfiguration：**


需要在配置类中添加@EnableAsync就可以使用多线程。在希望执行的并发方法中使用@Async就可以定义一个线程任务。通过spring给我们提供的ThreadPoolTaskExecutor就可以使用线程池。


	package com.wzy.springboot_multithread.Configuration;
	
	import org.springframework.context.annotation.Bean;
	import org.springframework.context.annotation.Configuration;
	import org.springframework.scheduling.annotation.EnableAsync;
	import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
	
	import java.util.concurrent.Executor;
	
	@Configuration
	@EnableAsync  // 启用异步任务
	public class AsyncConfiguration {
	
	    // 声明一个线程池(并指定线程池的名字)
	    @Bean("taskExecutor")
	    public Executor asyncExecutor() {
	        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
	        //核心线程数5：线程池创建时候初始化的线程数
	        executor.setCorePoolSize(5);
	        //最大线程数5：线程池最大的线程数，只有在缓冲队列满了之后才会申请超过核心线程数的线程
	        executor.setMaxPoolSize(5);
	        //缓冲队列500：用来缓冲执行任务的队列
	        executor.setQueueCapacity(500);
	        //允许线程的空闲时间60秒：当超过了核心线程出之外的线程在空闲时间到达之后会被销毁
	        executor.setKeepAliveSeconds(60);
	        //线程池名的前缀：设置好了之后可以方便我们定位处理任务所在的线程池
	        executor.setThreadNamePrefix("DailyAsync-");
	        executor.initialize();
	        return executor;
	    }
	}


有很多可以配置的东西。默认情况下，使用SimpleAsyncTaskExecutor



**CommonAutoConfiguration：**

需要配置RestTemplate相关configuration

	package com.wzy.springboot_multithread.Configuration;
	
	import org.springframework.context.annotation.Bean;
	import org.springframework.context.annotation.Configuration;
	import org.springframework.http.client.ClientHttpRequestFactory;
	import org.springframework.http.client.SimpleClientHttpRequestFactory;
	import org.springframework.web.client.RestTemplate;
	
	@Configuration
	public class CommonAutoConfiguration {
	
	    @Bean
	    public RestTemplate restTemplate(ClientHttpRequestFactory factory){
	        return new RestTemplate(factory);
	    }
	
	    @Bean
	    public ClientHttpRequestFactory simpleClientHttpRequestFactory(){
	        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
	        factory.setReadTimeout(5000);
	        factory.setConnectTimeout(5000);
	        return factory;
	    }
	
	}


**GitHubLookupService：**

用于提供服务。在定义了线程池之后，我们如何让异步调用的执行任务使用这个线程池中的资源来运行呢？方法非常简单，我们只需要在@Async注解中指定线程池名即可。


	package com.wzy.springboot_multithread.Service;
	
	import org.slf4j.Logger;
	import org.slf4j.LoggerFactory;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.scheduling.annotation.Async;
	import org.springframework.stereotype.Service;
	import org.springframework.web.client.RestTemplate;
	import java.util.concurrent.CompletableFuture;
	
	@Service
	public class GitHubLookupService {
	
	    private static final Logger logger = LoggerFactory.getLogger(GitHubLookupService.class);
	
	    @Autowired
	    private RestTemplate restTemplate;
	
	    // 这里进行标注为异步任务，在执行此方法的时候，会单独开启线程来执行(并指定线程池的名字)
	    @Async("taskExecutor")
	    public CompletableFuture<String> findUser(String user) throws InterruptedException {
	        logger.info("Looking up " + user);
	        String url = String.format("https://api.github.com/users/%s", user);
	        String results = restTemplate.getForObject(url, String.class);
	        // Artificial delay of 3s for demonstration purposes
	        Thread.sleep(3000L);
	        return CompletableFuture.completedFuture(results);
	    }
	}


findUser 方法被标记为Spring的 @Async 注解，表示它将在一个单独的线程上运行。该方法的返回类型是 CompleetableFuture 而不是 String，这是任何异步服务的要求。


**AsyncTests：**

测试类

	package com.wzy.springboot_multithread;
	
	import com.wzy.springboot_multithread.Service.GitHubLookupService;
	import org.junit.Test;
	import org.junit.runner.RunWith;
	import org.slf4j.Logger;
	import org.slf4j.LoggerFactory;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.boot.test.context.SpringBootTest;
	import org.springframework.test.context.junit4.SpringRunner;
	
	import java.util.concurrent.CompletableFuture;
	import java.util.concurrent.ExecutionException;
	
	@RunWith(SpringRunner.class)
	@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
	public class AsyncTests {
	    private static final Logger logger = LoggerFactory.getLogger(AsyncTests.class);
	
	    @Autowired
	    private GitHubLookupService gitHubLookupService;
	
	    @Test
	    public void asyncTest() throws InterruptedException, ExecutionException {
	        // Start the clock
	        long start = System.currentTimeMillis();
	
	        // Kick of multiple, asynchronous lookups
	        CompletableFuture<String> page1 = gitHubLookupService.findUser("PivotalSoftware");
	        CompletableFuture<String> page2 = gitHubLookupService.findUser("CloudFoundry");
	        CompletableFuture<String> page3 = gitHubLookupService.findUser("Spring-Projects");
	        CompletableFuture<String> page4 = gitHubLookupService.findUser("aaa");
	        CompletableFuture<String> page5 = gitHubLookupService.findUser("bbb");
	        CompletableFuture<String> page6 = gitHubLookupService.findUser("ccc");
	        CompletableFuture<String> page7 = gitHubLookupService.findUser("ddd");
	
	        System.out.println("----------------开始--------------------");
	
	        // Wait until they are all done
	        //join() 的作用：让“主线程”等待“子线程”结束之后才能继续运行
	        CompletableFuture.allOf(page1,page2,page3,page4,page5,page6,page7).join();
	
	
	        System.out.println("----------------结束--------------------");
	
	        // Print results, including elapsed time
	        float exc = (float)(System.currentTimeMillis() - start)/1000;
	        logger.info("Elapsed time: " + exc + " seconds");
	        logger.info("--> " + page1.get());
	        logger.info("--> " + page2.get());
	        logger.info("--> " + page3.get());
	        logger.info("--> " + page4.get());
	        logger.info("--> " + page5.get());
	        logger.info("--> " + page6.get());
	        logger.info("--> " + page7.get());
	    }
	
	}




## 验证

运行测试类后，得到：

![](../Images/2.png)


执行上面的单元测试，我们可以在控制台中看到所有输出的线程名前都是之前我们定义的线程池前缀名开始的，并且执行时间小于9秒，说明我们使用线程池来执行异步任务的试验成功了！

从线程的名字来看，只有5个线程，证明线程池设置成功。


#注意事项

在使用spring的异步多线程时经常回碰到多线程失效的问题，解决方式为：
异步方法和调用方法一定要写在不同的类中 ,如果写在一个类中,是没有效果的！
原因：

> spring对@Transactional注解时也有类似问题，spring扫描时具有@Transactional注解方法的类时，是生成一个代理类，由代理类去开启关闭事务，而在同一个类中，方法调用是在类体内执行的，spring无法截获这个方法调用。
> 