package com.wzy.dao;

import java.util.List;

import com.wzy.vo.Dept;
import org.apache.ibatis.annotations.Mapper;


@Mapper
public interface IDeptDAO {
	public boolean doCreate(Dept vo) ;
	public Dept findById(Long id) ;
	public List<Dept> findAll() ;
}
