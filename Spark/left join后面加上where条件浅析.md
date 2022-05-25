# left join后面加上where条件浅析


## 结论

where后面：是先连接然生成临时查询结果，然后再筛选

on后面：先根据条件过滤筛选，再连 生成临时查询结果


## 案例

	select a.*,b.*
	from table1 a
	left join table2 b on b.X=a.X
	where XXX
	
如上：一旦使用了left join，没有where条件时，左表table1会显示全部内容

　　  使用了where，只有满足where条件的记录才会显示（左表显示部分或者全部不显示）
　　  
　　  
## 原因分析

数据库在通过连接两张或多张表来返回记录时，都会生成一张中间的临时表，然后再将这张临时表返回给用户；

where条件是在临时表生成好后，再对临时表进行过滤的条件；

因此：where 条件加上，已经没有left join的含义（必须返回左边表的记录）了，条件不为真的就全部过滤掉。 


## 解决方案

1. where过滤结果作为子查询，和主表left，如下：

	select a.*,tmp.*
	from table1 a
	left join(
	    select a.*,b.*
	    from table1 a
	    left join table2 b on b.X=a.X
	    where XXX
	)tmp


2. 查询条件放在on后面

	select  a.*,b.*
	from  table1 a
	left  join  table2 b  on  b.X=a.X  and  XXX