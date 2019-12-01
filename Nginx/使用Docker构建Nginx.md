# 使用Docker构建Nginx


### 启动容器

	docker run --name nginx -p 2222:80 -d nginx
 
### 创建目录

	mkdir -p /root/nginx/{www,logs,conf,conf.d}

### 创建配置文件

	docker cp nginx:/etc/nginx/nginx.conf /root/nginx/conf

	docker cp nginx:/etc/nginx/conf.d/default.conf /root/nginx/conf.d/default.conf
 

### 删除容器

	docker stop nginx

	docker rm nginx


### 构建镜像

	docker run -d -p 2222:80 --name nginx -v /root/nginx/www:/usr/share/nginx/html -v /root/nginx/conf/nginx.conf:/etc/nginx/nginx.conf -v /root/nginx/logs:/var/log/nginx -v /root/nginx/conf.d/default.conf:/etc/nginx/conf.d/default.conf  nginx


### 构建测试页面

	cd /root/nginx/www
	
	vi index.html

	hello world!


## 测试

	http://47.112.142.231:2222/



	