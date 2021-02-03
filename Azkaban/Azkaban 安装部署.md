# Azkaban 安装步骤

参考资料： http://www.36nu.com/post/326

## 下载

	https://github.com/azkaban/azkaban/releases
	
## 安装

### 解压

	cd /root/azkaban_demo
	tar zxvf 3.90.0.tar.gz
	cd /root/azkaban_demo/azkaban-3.90.0
	
###	配置国内仓库
修改项目根目录下的文件 build.gradle

	buildscript {
	    repositories {
	        maven{ url 'http://maven.aliyun.com/nexus/content/groups/public/' }
	        maven{ url 'http://maven.aliyun.com/nexus/content/repositories/jcenter'}
	    }
	}
	
###  安装必要的库

	sudo yum install -y gcc-c++*

### 编译

	./gradlew build installDist -x test

