#ElasticSearch 文档映射


##动态映射

**动态映射**
![](../Images/6.png)

**支持的类型**
![](../Images/5.png)



##静态映射

	PUT /db_index/user/1
			{
				"name": "Jack",
				"sex": 1,
				"age": 25,
				"book": "Spring Boot 入门到精通",
				"remark": "hello world"
			}



## 对已存在的mapping映射进行修改	


		* 如果要推倒现有的映射, 你得重新建立一个静态索引
		
		* 然后把之前索引里的数据导入到新的索引里
		* 删除原创建的索引
		* 为新索引起个别名, 为原索引名
		
		POST _reindex
		{
		  "source": {
			"index": "db_index"
		  },
		  "dest": {
			"index": "db_index_2"
		  }
		}
		
		DELETE /db_index
		
		PUT /db_index_2/_alias/db_index



	其中第一步就把数据同时同步到db_index_2索引中了。




##keyword 与 text 映射类型的区别

	* 将 book 字段设置为 keyword 映射 （只能精准查询, 不能分词查询，能聚合、排序）
			POST /db_index/user/_search
			{
				"query": {
					"term": {
						"book": "Hadoop 入门到精通"
					}
				}
			}
			
		* 将 book 字段设置为 text 映射	（能模糊查询, 能分词查询，不能聚合、排序）
			POST /db_index/user/_search
			{
				"from": 0,
				"size": 2, 
				"query": {
					"match": {
						"book": "Hadoop"
					}
				}
			}