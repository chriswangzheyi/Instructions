# RbitMQ Docker搭建

## 搭建步骤：

###命令：

	docker pull rabbitmq (镜像未配有控制台)
	docker pull rabbitmq:management (镜像配有控制台)


### 启动

	docker run --name rabbitmq -d -p 15672:15672 -p 5672:5672 rabbitmq:management

## 验证

访问：

	http://47.112.142.231:15672/

账号：guest

密码：guest