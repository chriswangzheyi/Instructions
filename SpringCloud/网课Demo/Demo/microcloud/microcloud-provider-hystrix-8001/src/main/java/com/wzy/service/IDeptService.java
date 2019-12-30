package com.wzy.service;



import com.wzy.vo.Dept;

import java.util.List;


public interface IDeptService {
	public Dept get(long id) ;
	public boolean add(Dept dept) ;
	public List<Dept> list() ;
}
