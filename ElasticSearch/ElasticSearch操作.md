**ElasticSearch操作**


#1.基本操作

通过Kibana操作： Dev-Tools:

![](Images/3.png)


**创建索引**

PUT {index name}

例如： PUT /db_index


**获取索引**

GET {index name}

例如： GET /db_index


**删除索引**

DELETE {index name}

例如： DELETE /db_index


**添加文档**

 PUT /索引名称/类型/id

例如：

	PUT /db_index/user/1
				{
					"name": "Jack",
					"sex": 1,
					"age": 25,
					"book": "Spring Boot 入门到精通",
					"remark": "hello world"
				}
				
				PUT /db_index/user/2
				{
					"name": "Tom",
					"sex": 1,
					"age": 28,
					"book": "Spring Cloud 入门到精通",
					"remark": "hello java",
					"test": "add field"
				}
				
				PUT /db_index/user/3
				{
					"name": "Lily",
					"sex": 0,
					"age": 26,
					"book": "Hadoop 权威指南",
					"remark": "hello hadoop"
				}
				
				PUT /db_index/user/4
				{
					"name": "Colin",
					"sex": 0,
					"age": 22,
					"book": "Hadoop 技术内幕",
					"remark": "What is the hadoop"
				}
				
				PUT /db_index/user/5
				{
					"name": "Tobin",
					"sex": 0,
					"age": 19,
					"book": "Hadoop 入门到精通",
					"remark": "What is the hadoop"
				}			


**修改文档**

PUT /索引名称/类型/id

例子：

			PUT /db_index/user/1
			{
				"name": "JackTEST",
				"sex": 1,
				"age": 25,
				"book": "Spring Boot 入门到精通",
				"remark": "hello world"				
			}



**查询文档**

GET /索引名称/类型/id

例子：
	
    GET /db_index/user/1



**删除文档**

DELETE /索引名称/类型/id

例子：

	DELETE /db_index/user/1

---

#2.查询记录

**查看所有记录**

查询当前类型中的所有文档 _search 

		格式: GET /索引名称/类型/_search
		举例: GET /db_index/user/_search
		SQL:  select * from user


**按条件查询**

条件查询, 如要查询age等于28岁的 _search?q=*:***

		格式: GET /索引名称/类型/_search?q=*:***
		举例: GET /db_index/user/_search?q=age:28
		SQL:  select * from user where age = 28


**按范围条件查询**

范围查询, 如要查询age在25至26岁之间的 _search?q=***[** TO **]  注意: TO 必须为大写

		格式: GET /索引名称/类型/_search?q=***[25 TO 26]
		举例: GET /db_index/user/_search?q=age[25 TO 26]
		SQL:  select * from user where age between 25 and 26


**按ID批量查询**

根据多个ID进行批量查询 _mget

		格式: GET /索引名称/类型/_mget
		举例: GET /db_index/user/_mget
			  {
				  "ids":["1","2"]  
			  }
		SQL:  select * from user where id in (1,2)	


**按大于小于等于查询**
		
查询年龄小于等于28岁的 :<=

		格式: GET /索引名称/类型/_search?q=age:<=**
		举例: GET /db_index/user/_search?q=age:<=28
		SQL:  select * from user where age <= 28

查询年龄大于28前的 :>

		格式: GET /索引名称/类型/_search?q=age:>**
		举例: GET /db_index/user/_search?q=age:>28
		SQL:  select * from user where age > 28


**分页查询**


分页查询 from=*&size=*

		格式: GET /索引名称/类型/_search?q=age[25 TO 26]&from=0&size=1
		举例: GET /db_index/user/_search?q=age[25 TO 26]&from=0&size=1
		SQL:  select * from user where age between 25 and 26 limit 0, 1 

from 指定了页数， size指定了每一页的条数


**查询结果显示部分字段**

对查询结果只输出某些字段 _source=字段,字段

		格式: GET /索引名称/类型/_search?_source=字段,字段
		举例: GET /db_index/user/_search?_source=name,age
		SQL:  select name,age from user


**对结果排序**

对查询结果排序 sort=字段:desc/asc

		格式: GET /索引名称/类型/_search?sort=字段 desc
		举例: GET /db_index/user/_search?sort=age:desc
		SQL:  select * from user order by age desc


**多索引**

 多索引和多类别查询 GET {索引}
	
		在所有索引的所有类型中搜索
		/_search
		示例：GET /_search
		
		在索引gb的所有类型中搜索
		/gb/_search
		
		在索引gb和us的所有类型中搜索
		/gb,us/_search

		在以g或u开头的索引的所有类型中搜索
		/g*,u*/_search

		在索引gb的类型user中搜索
		/gb/user/_search
		
		在索引gb和us的类型为user和tweet中搜索
		/gb,us/user,tweet/_search

---

#DSL语言高级查询	

ES提供了强大的查询语言（DSL），它可以允许我们进行更加强大、复杂的查询。Elasticsearch DSL中有Query与Filter两种


##Query方式查询

会在ES中索引的数据都会存储一个_score分值，分值越高就代表越匹配。另外关于某个搜索的分值计算还是很复杂的，因此也需要一定的时间。

首先需要确定文件映射类型：

![](Images/4.png)

tpye为 keyword的词才可以用Query方式查询


错误示例：（因为term的类型不是keyword）
	

		根据名称精确查询姓名 term, term查询不会对字段进行分词查询，会采用精确匹配 
			注意: 采用term精确查询, 查询字段映射类型属于为keyword.
			举例: 
			POST /db_index/user/_search
			{
				"query": {
					"term": {
						"name": "Jack"
					}
				}
			}
			SQL: select * from user where name = 'Jack'


正确步骤：


 **删除原创建的索引**

			DELETE /db_index
			
** 创建索引**

			PUT /db_index
			
**设置文档映射**

			POST /db_index/_mapping/user
			{
				"user":{
					"properties":{
					   "name":{
							 "type":"keyword"
					   },
					   "sex":{
							"type":"integer"
					   },
					   "age":{
							"type":"integer"
					   },
					   "book":{
							"type":"text"
					   },
					   "remark":{
							"type":"text"
					   },
					   "test":{
							"type":"keyword"
					   }
					}
				}
			}	
	

**根据静态映射创建文档**

			PUT /db_index/user/1
			{
				"name": "Jack",
				"sex": 1,
				"age": 25,
				"book": "Spring Boot 入门到精通",
				"remark": "hello world"
			}
		
---	

## 查询语句


**精确查询**

term关键字为精确查询


			POST /db_index/user/_search
			{
				"query": {
					"term": {
						"name": "Jack"
					}
				}
			}




**模糊查询**

match关键字为模糊查询

	POST /db_index/user/_search
			{
				"from": 0,
				"size": 2, 
				"query": {
					"match": {
						"book": "Spring"
					}
				}
			}


---

**多字段模糊匹配查询与精准查询**

multi_match 关键字

	POST /db_index/user/_search
			{
				"query":{
					"multi_match":{
						"query":"Spring Hadoop",
						"fields":["book","remark"]
					}
				}
			}
			SQL: select * from user 
				 where book like '%Spring%' or book like '%Hadoop%' or remark like '%Spring%' or remark like '%Hadoop%'



**未指定字段条件查询 query_string , 含 AND 与 OR 条件**

	POST /db_index/user/_search
			{
				"query":{
					"query_string":{
						"query":"(Spring Cloud AND 入门到精通) OR Spring Boot"
					}
				}
			}


**指定字段条件查询 query_string , 含 AND 与 OR 条件**


	POST /db_index/user/_search
			{
				"query":{
					"query_string":{
						"query":"Spring Boot OR 入门到精通",
						"fields":["book","remark"]
					}
				}
			}


**范围查询**

			注：json请求字符串中部分字段的含义
			　　range：范围关键字
			　　gte 大于等于
			　　lte  小于等于
			　　gt 大于
			　　lt 小于
			　　now 当前时间

---

	POST /db_index/user/_search
				{
					"query" : {
						"range" : {
							"age" : {
								"gte":25,
								"lte":28
							}
						}
					}
				}
				SQL: select * from user where age between 25 and 28



**分页、输出字段、排序综合查询**

	POST /db_index/user/_search
			{
				"query" : {
					"range" : {
						"age" : {
							"gte":25,
							"lte":28
						}
					}
				},
				"from": 0,
				"size": 2,
				"_source": ["name", "age", "book"],
				"sort": {"age":"desc"}
			}


##Filter过滤器方式查询

查询不会计算相关性分值，也不会对结果进行排序, 因此效率会高一点，查询的结果可以被缓存。


**Filter Context 对数据进行过滤**


			POST /db_index/user/_search
			{
				"query" : {
					"bool" : {
						"filter" : {
							"term":{
								"age":25
							}
						}
					}
				}
			}		