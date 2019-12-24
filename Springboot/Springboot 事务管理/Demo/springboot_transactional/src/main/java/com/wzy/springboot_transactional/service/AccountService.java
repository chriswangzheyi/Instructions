package com.wzy.springboot_transactional.service;

import org.springframework.stereotype.Service;

@Service
public interface AccountService {

    //转账
    public void transfer(int outter,int inner,Integer money);
}
