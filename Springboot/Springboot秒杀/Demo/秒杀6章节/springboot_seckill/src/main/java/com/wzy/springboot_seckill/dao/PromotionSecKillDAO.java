package com.wzy.springboot_seckill.dao;

import com.wzy.springboot_seckill.entity.PromotionSecKill;

import java.util.List;

public interface PromotionSecKillDAO {
    List<PromotionSecKill> findUnstartSecKill();
    void update(PromotionSecKill ps);
    PromotionSecKill findById(Long psId);
    List<PromotionSecKill> findExpireSecKill();
}
