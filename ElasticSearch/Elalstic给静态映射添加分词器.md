#Elalstic给静态映射添加分词器

___

##操作步骤

创建静态映射时指定text类型的ik分词器

	
	 设置ik分词器的文档映射
			* 先删除之前的db_index

			 DELETE /db_index

			* 再创建新的db_index

			PUT /db_index

			* 定义ik_smart的映射
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
							"type":"text",
							"analyzer":"ik_smart",
							"search_analyzer":"ik_smart"
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



##验证

	* PUT前面的5条数据
			* 分词查询
			POST /db_index/user/_search
			{ 
				"query": {
					"match": {
						"book": "入"
					}
				}
			}	
			
			POST /db_index/user/_search
			{ 
				"query": {
					"match": {
						"book": "入门"
					}
				}
			}				

	