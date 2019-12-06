package com.wzy.springboot_seckill.dao;


import com.wzy.springboot_seckill.entity.GoodsCover;

import java.util.List;

public interface GoodsCoverDAO {
    public List<GoodsCover> findByGoodsId(Long goodsId);
}
