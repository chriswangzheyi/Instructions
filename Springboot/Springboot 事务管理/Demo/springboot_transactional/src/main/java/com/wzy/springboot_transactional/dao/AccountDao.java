package com.wzy.springboot_transactional.dao;

import org.apache.ibatis.annotations.Param;

public interface AccountDao {
    public void moveIn(@Param("id") int id, @Param("money") float money); // 转入

    public void moveOut(@Param("id") int id, @Param("money") float money); // 转出

}
