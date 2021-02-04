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

在${USER_HOME}/.gradle/下创建init.gradle文件

	allprojects{
	    repositories {
	        def ALIYUN_REPOSITORY_URL = 'https://maven.aliyun.com/repository/public/'
	        def ALIYUN_JCENTER_URL = 'https://maven.aliyun.com/repository/jcenter/'
	        def ALIYUN_GOOGLE_URL = 'https://maven.aliyun.com/repository/google/'
	        def ALIYUN_GRADLE_PLUGIN_URL = 'https://maven.aliyun.com/repository/gradle-plugin/'
	        all { ArtifactRepository repo ->
	            if(repo instanceof MavenArtifactRepository){
	                def url = repo.url.toString()
	                if (url.startsWith('https://repo1.maven.org/maven2/')) {
	                    project.logger.lifecycle "Repository ${repo.url} replaced by $ALIYUN_REPOSITORY_URL."
	                    remove repo
	                }
	                if (url.startsWith('https://jcenter.bintray.com/')) {
	                    project.logger.lifecycle "Repository ${repo.url} replaced by $ALIYUN_JCENTER_URL."
	                    remove repo
	                }
	                if (url.startsWith('https://dl.google.com/dl/android/maven2/')) {
	                    project.logger.lifecycle "Repository ${repo.url} replaced by $ALIYUN_GOOGLE_URL."
	                    remove repo
	                }
	                if (url.startsWith('https://plugins.gradle.org/m2/')) {
	                    project.logger.lifecycle "Repository ${repo.url} replaced by $ALIYUN_GRADLE_PLUGIN_URL."
	                    remove repo
	                }
	            }
	        }
	        maven { url ALIYUN_REPOSITORY_URL }
	        maven { url ALIYUN_JCENTER_URL }
	        maven { url ALIYUN_GOOGLE_URL }
	        maven { url ALIYUN_GRADLE_PLUGIN_URL }
	    }
	}


###  安装必要的库

	sudo yum install -y gcc-c++*

### 编译

	./gradlew build installDist -x test

