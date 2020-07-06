# Centos 安装 Docker

## 步骤

	#更新yum包
	yum update

	#安装需要的软件包
	yum install -y yum-utils device-mapper-persistent-data lvm2
	
	#设置yum源
	yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

	#查看所有仓库中所有docker版本，并选择特定版本安装
	yum list docker-ce --showduplicates | sort -r

	#安装Docker，命令：yum install docker-ce-版本号,这里我选择了18.06.3.ce-3.el7
	yum install docker-ce-18.06.3.ce-3.el7

	#启动Docker，命令：systemctl start docker，然后加入开机启动
	systemctl start docker
	systemctl enable docker

	#验证是否安装成功
	docker version

	#使用Docker 中国加速器
	vi  /etc/docker/daemon.json
	#添加后：
	{
	    "registry-mirrors": ["https://registry.docker-cn.com"],
	    "live-restore": true
	}

	
	
或者选择DaoCloud镜像站点安装 https://www.daocloud.io/mirror#accelerator-doc

	curl -sSL https://get.daocloud.io/daotools/set_mirror.sh | sh -s http://f1361db2.m.daocloud.io

	#重启docker服务

	systemctl daemon-reload
	systemctl restart docker


## 安装上传下载插件

	yum -y install lrzsz
	

## 安装Docker compose

从github中下载安装包

	https://github.com/docker/compose/releases

将docker-compose上传至服务器。

	rz

将文件传到对应目录

	cp /root/docker-compose-Linux-x86_64 /usr/local/bin/docker-compose


增加权限

	chmod +x /usr/local/bin/docker-compose

验证

	docker-compose --version