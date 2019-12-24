package com.wzy.springboot_transactional.impl;

import com.wzy.springboot_transactional.dao.AccountDao;
import com.wzy.springboot_transactional.service.AccountService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AccountServiceImpl implements AccountService {

    @Autowired(required=false)
    private AccountDao accountDao;

    // 放开注解后，改方法实现事务控制
    //@Transactional
    public void transfer(int outter, int inner, Integer money) {

        accountDao.moveOut(outter, money); //转出

        //int i = 1/0;  // 抛出异常，模拟程序出错

        accountDao.moveIn(inner, money); //转入

    }
}
