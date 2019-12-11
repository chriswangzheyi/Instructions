package com.wzy.springboot_seckill.controller;

import com.wzy.springboot_seckill.entity.Order;
import com.wzy.springboot_seckill.service.PromotionSecKillService;
import com.wzy.springboot_seckill.service.exception.SecKillException;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.servlet.ModelAndView;
import javax.annotation.Resource;
import java.util.HashMap;
import java.util.Map;

@Controller
public class SecKillController {

    @Resource
    PromotionSecKillService promotionSecKillService;

    //获取抢购资格
    //localhost:8080\seckill?psid=1&userid=user001
    @RequestMapping("/seckill")
    @ResponseBody
    public Map processSecKill(Long psid , String userid){

        Map result = new HashMap();

        try {
            promotionSecKillService.processSecKill(psid , userid , 1);
            String orderNo = promotionSecKillService.sendOrderToQueue(userid);
            Map data = new HashMap();
            data.put("orderNo", orderNo);
            result.put("code", "0"); //code0表示操作成功
            result.put("message", "success");
            result.put("data", data);
        } catch (SecKillException e) {
            result.put("code", "500");
            result.put("message", e.getMessage());
        }
        return result;
    }

    //查询订单详情
    //http://localhost:8080/checkorder?orderNo=df79877c-85f1-4305-8f95-72572381d0de
    @GetMapping("/checkorder")
    public ModelAndView checkOrder(String orderNo){
        Order order =  promotionSecKillService.checkOrder(orderNo);
        ModelAndView mav = new ModelAndView();
        if(order != null){
            mav.addObject("order", order);
            mav.setViewName("/order");
        }else{
            mav.addObject("orderNo", orderNo);
            mav.setViewName("/waiting");
        }
        return mav;
    }
}
