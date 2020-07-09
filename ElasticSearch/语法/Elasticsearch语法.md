# Elasticsearch语法


可以使用Kibana dev tool

## 新增

	POST /{index}/{type}/{id}
	{
	  "field": "value",
	  ...
	}


例子：

	POST twitter/_doc/1
	{
	    "username":"Dannis",
	    "uid":1
	}

## 查询

### 返回所有文档

	GET /_search
	{}


### 获取单个文档

GET /{index}/{type}/{id}

	GET twitter/_doc/1



## 修改


	POST twitter/_doc/1
	{
	    "username":"wangzheyi",
	    "uid":1
	}

## 删除

	DELETE /{index}/{type}/{id}

例子：

	DELETE twitter/_doc/1