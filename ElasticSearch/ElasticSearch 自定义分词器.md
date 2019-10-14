#ElasticSearch自定义分词器词


---

##操作步骤

进入容器

	docker exec -it myes bash

安装vi

	apt-get install vim
	apt-get update
	apt-get install vim


变更路径

	cd /usr/share/elasticsearch/plugins/ik/config

创建文件夹并进入

	mkdir custom
	cd custom

添加文件

	vi my_word.dic

添加词汇如流浪地球

修改配置文件

	vi /usr/share/elasticsearch/plugins/ik/config/IKAnalyzer.cfg.xml
	
添加内容
	
	<entry key="ext_dict">custom/my_word.dic</entry>

重启容器
	
 docker restart myes

验证

	进入kibana

		GET _analyze
		{
			"analyzer": "ik_smart",
			"text": "流浪地球"
		}		