# Dokcer 搭建OpentsDB


## 启动容器

	docker run -d -p 4242:4242 --name opentsdb petergrace/opentsdb-docker

## 进入容器 （非必须）

	docker exec -it opentsdb /bin/bash


## 访问页面

	http://47.112.142.231:4242/