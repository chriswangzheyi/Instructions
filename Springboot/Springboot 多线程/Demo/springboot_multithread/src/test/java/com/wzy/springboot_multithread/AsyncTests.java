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


