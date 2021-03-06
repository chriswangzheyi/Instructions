package com.wzy.springboot_seckill.scheduler;

import com.wzy.springboot_seckill.dao.PromotionSecKillDAO;
import com.wzy.springboot_seckill.entity.PromotionSecKill;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import javax.annotation.Resource;
import java.util.List;

@Component
public class SecKillTask {
    @Resource
    private PromotionSecKillDAO promotionSecKillDAO;
    //RedisTempldate是Spring封装的Redis操作类，提供了一系列操作redis的模板方法
    @Resource
    private RedisTemplate redisTemplate;

    //每隔5秒执行一次
    @Scheduled(cron = "0/5 * * * * ?")
    public void startSecKill(){
        //根据起始时间及状态码（0为未启动）查询到时间却未启动的任务
        List<PromotionSecKill> list  = promotionSecKillDAO.findUnstartSecKill();

        for(PromotionSecKill ps : list){
            System.out.println(ps.getPsId() + "秒杀活动已启动");
            //删掉以前重复的活动任务缓存
            redisTemplate.delete("seckill:count:" + ps.getPsId());
            //有几个库存商品，则初始化几个list对象
            for(int i = 0 ; i < ps.getPsCount() ; i++){
                redisTemplate.opsForList().rightPush("seckill:count:" + ps.getPsId(), ps.getGoodsId().toString());
            }
            //将任务状态设置为1（启动）
            ps.setStatus(1);
            promotionSecKillDAO.update(ps);
        }
    }

    @Scheduled(cron = "0/5 * * * * ?")
    public void endSecKill(){
        List<PromotionSecKill> psList = promotionSecKillDAO.findExpireSecKill();
        for (PromotionSecKill ps : psList) {
            System.out.println(ps.getPsId() + "秒杀活动已结束");
            ps.setStatus(2);
            promotionSecKillDAO.update(ps);
            redisTemplate.delete("seckill:count:" + ps.getPsId());
        }
    }
}
