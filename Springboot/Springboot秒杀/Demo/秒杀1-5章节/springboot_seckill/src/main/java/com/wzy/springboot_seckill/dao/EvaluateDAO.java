package com.wzy.springboot_seckill.dao;


import com.wzy.springboot_seckill.entity.Evaluate;

import java.util.List;

public interface EvaluateDAO {
    public List<Evaluate> findByGoodsId(Long goodsId);
}
