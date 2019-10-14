#ElasticSearch Head插件安装

head插件是将es图形化展现给用户，它是集群管理、数据可视化、增删改查、查询语句可视化工具。

参考信息：

> https://cloud.tencent.com/developer/article/1368381


##安装步骤

进入容器

	docker exec -it myes bash

更新apt-get

	apt-get update

安装unzip

	apt-get install zip

安装vim

	apt-get install vim


安装nodejs

	wget https://npm.taobao.org/mirrors/node/latest-v4.x/node-v4.4.7-linux-x64.tar.gz

	tar -zxvf node-v4.4.7-linux-x64.tar.gz

修改配置文件

	vim /etc/profile

---

	export NODE_HOME=/usr/share/elasticsearch/node-v4.4.7-linux-x64
	export PATH=$PATH:$NODE_HOME/bin

source /etc/profile

确定是否配置成功

	echo $NODE_HOME

更换目录

	cd /usr/share/elasticsearch

因为head插件不能放在plugins目录下


下载并安装head插件

	wget  https://github.com/mobz/elasticsearch-head/archive/master.zip
	unzip master.zip
	rm master.zip

安装nodejs相关安装包

	cd /usr/share/elasticsearch/elasticsearch-head-master

	npm install -g grunt-cli

	npm install grunt --save-dev
	#确认是否成功
	grunt -version



**修改两处的配置**


安装ElasticSearch目录下的config/elasticsearch.yml

vim /usr/share/elasticsearch/config/elasticsearch.yml
	
	 # 增加如下字段
	http.cors.enabled: true
	http.cors.allow-origin: "*"


修改elasticsearch-head下的Gruntfile.js

vim /usr/share/elasticsearch/elasticsearch-head-master/Gruntfile.js

增加

	hostname: '0.0.0.0',

![](../Images/11.png)


完成安装：

	cd /usr/share/elasticsearch/elasticsearch-head-master
	npm install


重启Docker

重启后进入容器

	docker exec -it myes bash

	source /etc/profile

	cd /usr/share/elasticsearch/elasticsearch-head-master	

	grunt server



##验证

进入 {宿主机Ip}:9100 端口查看

![](../Images/12.png)



