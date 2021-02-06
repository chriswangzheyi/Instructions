# Nacos Docker安装

## 安装

	docker pull nacos/nacos-server

## 启动

	docker run -d -p 8848:8848 --env MODE=standalone  --name nacos  nacos/nacos-server

## 验证是否成功

	http://47.112.142.231:8848/nacos

账号密码都是nacos



