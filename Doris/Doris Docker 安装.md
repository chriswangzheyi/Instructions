# Doris Docker 安装

参考：https://doris.apache.org/community/developer-guide/docker-dev/


## dockerfi

	vim Dockerfile

插入

	FROM apache/incubator-doris:build-env-latest
	
	USER root
	WORKDIR /root
	RUN echo '<!!! root password !!!>' | passwd root --stdin
	
	RUN yum install -y vim net-tools man wget git mysql lsof bash-completion \
	        && cp /var/local/thirdparty/installed/bin/thrift /usr/bin
	
	# safer usage, create new user instead of using root
	RUN yum install -y sudo \
	        && useradd -ms /bin/bash <!!! your user !!!> && echo <!!! your user password !!!> | passwd <!!! your user !!!> --stdin \
	        && usermod -a -G wheel <!!! your user !!!>
	
	USER <!!! your user !!!>
	WORKDIR /home/<!!! your user !!!>
	RUN git config --global color.ui true \
	        && git config --global user.email "<!!! your git email !!!>" \
	        && git config --global user.name "<!!! your git username !!!>"
	
	# install zsh and oh my zsh, easier to use on, you can remove it if you don't need it
	USER root
	RUN yum install -y zsh \
	        && chsh -s /bin/zsh <!!! your user !!!>
	USER <!!! your user !!!>
	RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh \
	        && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
	        && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting


## 制作镜像

	docker build -t doris .

## 启动镜像

	docker run -it doris:latest /bin/bash
	

## 创建文件并下载doris

	cd /Users/zheyiwang/Downloads/doris
	mkdir code && cd code
	
## 编译	
	git clone https://github.com/apache/doris.git
	sh build.sh --clean --be --fe --ui
	sh build.sh

## 启动

### 启动FE

	mkdir meta_dir
	cd output/fe
	sh bin/start_fe.sh --daemon
	
### 启动BE

	cd output/be
	sh bin/start_be.sh --daemon

### 连接

	mysql -h 127.0.0.1 -P 9030 -u root