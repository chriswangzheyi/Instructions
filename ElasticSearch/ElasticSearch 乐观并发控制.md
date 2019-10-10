#ElasticSearch 乐观并发控制


![](../Images/7.png)

通过在请求中加入version来控制


	再以创建一个文档为例

		PUT /db_index/user/1
		{
			"name": "Jack",
			"sex": 1,
			"age": 25,
			"book": "Spring Boot 入门到精通",
			"remark": "hello world"
		}
	
	实现_version乐观锁更新文档

		PUT /db_index/user/1?version=1
		{
			"name": "Jack",
			"sex": 1,
			"age": 25,
			"book": "Spring Boot 入门到精通",
			"remark": "hello world"
		}