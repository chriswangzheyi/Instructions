# HBase 基本语法

参考： https://blog.csdn.net/kangkangwanwan/article/details/60960731

## 查询

### list 

查看有哪些表

### list  表名   

列出关于你的表的信息
	
	hbase(main):012:0* list 'test'
	TABLE                                                                                                                                                                 
	test                                                                                                                                                                  
	1 row(s)
	Took 0.0305 seconds                                                                                                                                                   
	=> ["test"]


### scan 表名   

查看表中的所有数据

	hbase(main):007:0> scan 'test'
	ROW                                        COLUMN+CELL                                                                                                                
	 row1                                      column=cf:a, timestamp=1589867231785, value=value1                                                                         
	 row2                                      column=cf:b, timestamp=1589867236494, value=value2                                                                         
	 row3                                      column=cf:c, timestamp=1589867241960, value=value3           


### scan '表名',{STARTROW => '起始row名',STOPROW => '终止row名'}

按起始条件搜索

	hbase(main):009:0> scan 'test',{STARTROW => 'row1',STOPROW => 'row1'}
	ROW                                        COLUMN+CELL                                                                                                                
	 row1                                      column=cf:a, timestamp=1589867231785, value=value1                                                                         
	1 row(s)
	Took 0.0430 seconds    

或者
                    
	base(main):020:0* scan 'test',{STARTROW => 'row1',STOPROW => 'row3'}
	ROW                                        COLUMN+CELL                                                                                                                
	 row1                                      column=cf:a, timestamp=1589867231785, value=value1                                                                         
	 row2                                      column=cf:b, timestamp=1589867236494, value=value2                                                                         
	2 row(s)
                                                        



###  scan '表名',{STARTROW => '起始row'}

	hbase(main):021:0> scan 'test',{STARTROW => 'row1'}
	ROW                                        COLUMN+CELL                                                                                                                
	 row1                                      column=cf:a, timestamp=1589867231785, value=value1                                                                         
	 row2                                      column=cf:b, timestamp=1589867236494, value=value2                                                                         
	 row3                                      column=cf:c, timestamp=1589867241960, value=value3                                                                         
	3 row(s)



### get 'table_name', 'rowkey' 

 按照表名、row名获取单行的数据
	
	get 'test','row1

### get 表名，row名，列族名

 按照表名、row名、族列名获取单行的数据

	get 'test','row1','cf:d'

## 建表
	
	create 'namespace:table_name, 'family1','family2','familyN'    

创建表和列簇
	
	create 'test', 'cf'  #创建列族在default命名空间中
	create 'ns1:test','cf'  #创建列族在ns1命名空间中


把数据放到表中：

	put 'table_name', 'rowkey', 'family:column', 'value' 

	
	hbase(main):004:0* put 'test','row1','cf:a','value1'
	Took 0.2791 seconds                                                                                                                                                   
	hbase(main):005:0> put 'test','row2','cf:b','value2'
	Took 0.0556 seconds                                                                                                                                                   
	hbase(main):006:0> put 'test','row3','cf:c','value3'
	Took 0.0232 seconds 


### put 表名 row名 列族名 信息

也可以用put更新表

	hbase(main):023:0> put 'test','1001','cf:d','value4'
                                                                                                                                                 
	hbase(main):024:0> get 'test','1001'
	COLUMN                                     CELL                                                                                                                       
	 cf:d                                      timestamp=1589867796837, value=value4                                                                                      
	1 row(s)
                                                                                                                                                  
	hbase(main):025:0> put 'test','1001','cf:d','value5'
                                                                                                                                                 
	hbase(main):026:0> get 'test','1001'
	COLUMN                                     CELL                                                                                                                       
	 cf:d                                      timestamp=1589867850988, value=value5                                                                                      
	1 row(s)
         
## 删除

### drop 表名

删除表

	drop 'test'

### deleteall 表名，row名

按row删除内容

	deleteall  'test','0001'

### deleteall 表名，row名，列簇名

	deleteall  'test','0001','cf:e'



## 统计

### count 表名

统计表数据行数

	count 'test'


## 创建namespace

	create_namespace "name"

hbase中没有数据库概念，hbase中有namespace相当于hive中的数据库。

## 查看namespace列表

	list_namespace


## 综合例子

![](Images/1.png)

建表语句：

	create 'userinfo', 'personal','office'
	
	put 'userinfo','Row1','personal:name','张三'
	put 'userinfo','Row1','personal:city','北京'
	put 'userinfo','Row1','personal:phone','1311111111'
	put 'userinfo','Row1','office:tel','010-11111111'
	put 'userinfo','Row1','office:address','帝都大厦-18F-01'




[]()