package com.wzy.springboot_seckill.dao;


import com.wzy.springboot_seckill.entity.GoodsDetail;

import java.util.List;

public interface GoodsDetailDAO {
    public List<GoodsDetail> findByGoodsId(Long goodsId);
}
