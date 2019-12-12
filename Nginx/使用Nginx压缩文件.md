# 使用Nginx压缩文件

---

## 原理

![](../Images/2.png)

## 什么样的资源不适合开启gzip压缩？

二进制资源：例如图片/mp3这样的二进制文件,不必压缩；因为压缩率比较小, 比如100->80字节,而且压缩也是耗费CPU资源的.


## 修改配置文件


vi /root/nginx/conf/nginx.conf 

原始文件如下：


	user  nginx;
	worker_processes  1;
	
	error_log  /var/log/nginx/error.log warn;
	pid        /var/run/nginx.pid;
	
	
	events {
	    worker_connections  1024;
	}
	
	
	http {
	    include       /etc/nginx/mime.types;
	    default_type  application/octet-stream;
	
	    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
	                      '$status $body_bytes_sent "$http_referer" '
	                      '"$http_user_agent" "$http_x_forwarded_for"';
	
	    access_log  /var/log/nginx/access.log  main;
	
	    sendfile        on;
	    #tcp_nopush     on;
	
	    keepalive_timeout  65;
	
	    #gzip  on;
	
	    include /etc/nginx/conf.d/*.conf;
	}



## gzip配置的常用参数

1. gzip on|off; #是否开启gzip
1. gzip_buffers 32 4K| 16 8K #缓冲(压缩在内存中缓冲几块? 每块多大?) 
1. gzip_comp_level [1-9] #推荐6 压缩级别(级别越高,压的越小,越浪费CPU计算资源)
1. gzip_disable #正则匹配UA 什么样的Uri不进行gzip
1. gzip_min_length 200 # 开始压缩的最小长度(再小就不要压缩了,意义不在)
1. gzip_http_version 1.0|1.1 # 开始压缩的http协议版本(可以不设置,目前几乎全是1.1协议)
1. gzip_proxied # 设置请求者代理服务器,该如何缓存内容
1. gzip_types text/plain application/xml # 对哪些类型的文件用压缩 如txt,xml,html ,css
1. gzip_vary on|off # 是否传输gzip压缩标志


## 修改后的配置文件


	user  nginx;
	worker_processes  1;
	
	error_log  /var/log/nginx/error.log warn;
	pid        /var/run/nginx.pid;
	
	
	events {
	    worker_connections  1024;
	}
	
	
	http {
	    include       /etc/nginx/mime.types;
	    default_type  application/octet-stream;
	
	    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
	                      '$status $body_bytes_sent "$http_referer" '
	                      '"$http_user_agent" "$http_x_forwarded_for"';
	
	    access_log  /var/log/nginx/access.log  main;
	
	    sendfile        on;
	    #tcp_nopush     on;
	
	    keepalive_timeout  65;
	
		gzip on;
		gzip_min_length 1k;
		gzip_buffers 4 16k;
		#gzip_http_version 1.0;
		gzip_comp_level 2;
		gzip_types text/plain application/x-javascript text/css application/xml text/javascript application/x-httpd-php image/jpeg image/gif image/png;
		gzip_vary off;
		gzip_disable "MSIE [1-6]\.";
	
	
	    include /etc/nginx/conf.d/*.conf;
	}



## 比较效果

	#重启Nginx
	docker restart nginx


![](../Images/3.png)


	#页面压缩测试
	curl -I -H "Accept-Encoding: gzip, deflate" "http://www.slyar.com/blog/"
	
	#css压缩测试
	curl -I -H "Accept-Encoding: gzip, deflate" "http://www.slyar.com/blog/wp-content/plugins/photonic/include/css/photonic.css"
 

