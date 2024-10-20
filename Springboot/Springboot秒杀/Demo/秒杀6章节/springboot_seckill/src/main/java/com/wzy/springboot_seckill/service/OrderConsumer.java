package com.wzy.springboot_seckill.service;

import com.rabbitmq.client.Channel;
import com.wzy.springboot_seckill.dao.OrderDAO;
import com.wzy.springboot_seckill.entity.Order;
import org.springframework.amqp.rabbit.annotation.*;
import org.springframework.amqp.support.AmqpHeaders;
import org.springframework.messaging.handler.annotation.Headers;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Component;
import javax.annotation.Resource;
import java.io.IOException;
import java.util.Date;
import java.util.Map;

@Component
public class OrderConsumer {
    @Resource
    private OrderDAO orderDAO;

    @RabbitListener(
            bindings = @QueueBinding(
                    value = @Queue(value = "queue-order") ,
                    exchange = @Exchange(value = "exchange-order" , type = "fanout")
            )
    )

    @RabbitHandler
    public void handleMessage(@Payload Map data , Channel channel ,
                              @Headers Map<String,Object> headers){
        System.out.println("=======获取到订单数据:" + data + "===========);");

        try {
            //模拟对接支付宝、对接物流系统、日志登记。。。。
            try {
                Thread.sleep(10000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            Order order = new Order();
            order.setOrderNo(data.get("orderNo").toString());
            order.setOrderStatus(0);
            order.setUserid(data.get("userid").toString());
            order.setRecvName("xxx");
            order.setRecvMobile("1393310xxxx");
            order.setRecvAddress("xxxxxxxxxx");
            order.setAmout(19.8f);
            order.setPostage(0f);
            order.setCreateTime(new Date());
            orderDAO.insert(order);
            Long tag = (Long)headers.get(AmqpHeaders.DELIVERY_TAG);
            channel.basicAck(tag , false);//消息确认
            System.out.println(data.get("orderNo") + "订单已创建");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
