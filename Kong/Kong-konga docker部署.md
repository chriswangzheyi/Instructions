# Kong/konga docker部署

参考：https://zhuanlan.zhihu.com/p/421894560

## 介绍

Kong主要有三个组件：

* Kong Server ：基于nginx的服务器，用来接收API请求。
* Apache Cassandra/PostgreSQL ：用来存储操作数据。
* Kong dashboard：官方推荐UI管理工具，当然，也可以使用 restfull 方式 管理admin api。

## Kong的主要功能

* 高级路由、负载平衡、健康检查——所有这些都可以通过管理 API 或声明性配置进行配置。
* 使用 JWT、基本身份验证、ACL 等方法对API 进行身份验证和授权。
* 代理、SSL/TLS 终止以及对 L4 或 L7 流量的连接支持。
* 用于实施流量控制、req/res转换、日志记录、监控和包括插件开发人员中心的插件。
* 复杂的部署模型，如声明式无数据库部署和混合部署（控制平面/数据平面分离），没有任何供应商锁定。
* 本机入口控制器支持服务Kubernetes。

## 安装

### Kong

  $ git clone https://github.com/Kong/docker-kong
  $ cd compose/
  $ docker-compose up
  
### Konga

	$ git clone https://github.com/pantsel/konga.git
	$ cd konga
	$ npm run postinstall && npm i
	
生成配置文件

	PORT=1337
	NODE_ENV=production
	KONGA_HOOK_TIMEOUT=120000
	DB_ADAPTER=mysql
	DB_HOST=127.0.0.1
	DB_PORT=3306
	DB_USER=konga
	DB_DATABASE=konga
	DB_PASSWORD=konga
	KONGA_LOG_LEVEL=warn
	TOKEN_SECRET=some_secret_token
	
### 执行数据库迁移

	node ./bin/konga.js  prepare --adapter mysql --uri mysql://localhost:5432/konga
	
### 启动 Konga

	npm run production
	

## 保持 Konga 在后台运行

	pm2 start app.js --name konga