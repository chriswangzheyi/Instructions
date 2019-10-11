#ElasticSearch 中文分词



拆词语法：


		GET _analyze
		{
			"text": "内容"
		}


例如：

		GET _analyze
		{
			"text": "我爱中国"
		}

拆分结果：

	{
	  "tokens": [
	    {
	      "token": "我",
	      "start_offset": 0,
	      "end_offset": 1,
	      "type": "<IDEOGRAPHIC>",
	      "position": 0
	    },
	    {
	      "token": "爱",
	      "start_offset": 1,
	      "end_offset": 2,
	      "type": "<IDEOGRAPHIC>",
	      "position": 1
	    },
	    {
	      "token": "中",
	      "start_offset": 2,
	      "end_offset": 3,
	      "type": "<IDEOGRAPHIC>",
	      "position": 2
	    },
	    {
	      "token": "国",
	      "start_offset": 3,
	      "end_offset": 4,
	      "type": "<IDEOGRAPHIC>",
	      "position": 3
	    }
	  ]
	}



中国被当成了两个词


##在Docker中安装分词插件


查看Elasticsearch版本

	curl -XGET localhost:9200

![](../Images/9.png)


在github上看到

	https://github.com/medcl/elasticsearch-analysis-ik/releases?after=v6.3.2

![](../Images/10.png)

可以看到有版本一致的下载包


---

进入es容器 

	docker exec -it myes bash


安装unzip

	apt-get install zip




**进入plugins目录后:**


下载

	wget https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v5.6.12/elasticsearch-analysis-ik-5.6.12.zip


解压
	mv elasticsearch-analysis-ik-5.6.12.zip ik.zip

	unzip ik.zip


删除源文件

	rm -rf ik.zip

退出容器后重启elasticSearch容器

	docker restart myes



##验证

在kibana 中请求：


	GET _analyze
		{
			"analyzer": "ik_smart",
			"text": "我爱中国"
		}


结果：

	{
	  "tokens": [
	    {
	      "token": "我",
	      "start_offset": 0,
	      "end_offset": 1,
	      "type": "CN_CHAR",
	      "position": 0
	    },
	    {
	      "token": "爱",
	      "start_offset": 1,
	      "end_offset": 2,
	      "type": "CN_CHAR",
	      "position": 1
	    },
	    {
	      "token": "中国",
	      "start_offset": 2,
	      "end_offset": 4,
	      "type": "CN_WORD",
	      "position": 2
	    }
	  ]
	}