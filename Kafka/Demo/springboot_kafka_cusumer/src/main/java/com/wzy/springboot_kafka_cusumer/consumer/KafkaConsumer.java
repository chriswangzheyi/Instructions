package com.wzy.springboot_kafka_cusumer.consumer;

import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

/**
 * 消费者
 * 使用@KafkaListener注解,可以指定:主题,分区,消费组
 */
@Component
public class KafkaConsumer {

    @KafkaListener(topics = {"test"})
    public void receive(String message){
        System.out.println("test--消费消息:" + message);
    }
}