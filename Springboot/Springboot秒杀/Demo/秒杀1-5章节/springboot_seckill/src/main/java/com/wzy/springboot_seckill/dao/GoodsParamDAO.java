package com.wzy.springboot_seckill.dao;


import com.wzy.springboot_seckill.entity.GoodsParam;

import java.util.List;

public interface GoodsParamDAO {
    public List<GoodsParam> findByGoodsId(Long goodsId);
}
