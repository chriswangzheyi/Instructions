package com.wzy.springboot_seckill.dao;

import com.wzy.springboot_seckill.entity.Order;

public interface OrderDAO {
    public void insert(Order order);

    public Order findByOrderNo(String orderNo);
}
