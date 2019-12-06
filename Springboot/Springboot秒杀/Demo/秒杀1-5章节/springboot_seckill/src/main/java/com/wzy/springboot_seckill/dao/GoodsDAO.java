package com.wzy.springboot_seckill.dao;


import com.wzy.springboot_seckill.entity.Goods;

import java.util.List;

public interface GoodsDAO {
    public Goods findById(Long goodsId);
    public List<Goods> findAll();
    public List<Goods> findLast5M();
}
