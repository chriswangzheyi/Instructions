package com.wzy.service.impl;

import java.util.List;

import javax.annotation.Resource;

import com.wzy.dao.IDeptDAO;
import com.wzy.service.IDeptService;
import com.wzy.vo.Dept;
import org.springframework.stereotype.Service;


@Service
public class DeptServiceImpl implements IDeptService {
	@Resource
	private IDeptDAO deptDAO ;
	@Override
	public Dept get(long id) {
		return this.deptDAO.findById(id);
	}

	@Override
	public boolean add(Dept dept) {
		return this.deptDAO.doCreate(dept);
	}

	@Override
	public List<Dept> list() {
		return this.deptDAO.findAll();
	}

}
