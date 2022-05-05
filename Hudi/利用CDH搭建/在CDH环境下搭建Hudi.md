# 在CDH环境下搭建Hudi

## 前置条件

### 环境

CDH 6.3.2


## 安装步骤

### 安装maven

	tar -zxvf apache-maven-3.6.1-bin.tar.gz -C /data/module/
	
	mv apache-maven-3.6.1 maven
	
修改配置文件	

	vi /etc/profile
	
	#MAVEN_HOME
	export MAVEN_HOME=/data/module/maven
	export PATH=$PATH:$MAVEN_HOME/bin
	
使得配置生效
	
	source /etc/profile  
	mvn -v 	
	
修改setting.xml，指定为阿里云

	vim maven/conf/settings.xml

	<!-- 添加阿里云镜像-->
	<mirror>
	        <id>nexus-aliyun</id>
	        <mirrorOf>central</mirrorOf>
	        <name>Nexus aliyun</name>
	        <url>http://maven.aliyun.com/nexus/content/groups/public</url>
	</mirror>


修改setting.xml,指定路径

	<localRepository>/home/maven/repo</localRepository>

### 安装git

	yum install git
	git --version

### 安装Hudi

###下载源码

	git clone --branch release-0.9.0 https://gitee.com/apache/Hudi.git
	
### 编译


	mvn clean package -DskipTests

由于cdh使用的spark 版本是 2.11_2.4.0 所以选择hudi pom里自带的2.4.4


	[INFO] Reactor Summary for Hudi 0.9.0:
	[INFO] 
	[INFO] Hudi ............................................... SUCCESS [  1.844 s]
	[INFO] hudi-common ........................................ SUCCESS [ 13.048 s]
	[INFO] hudi-timeline-service .............................. SUCCESS [  2.548 s]
	[INFO] hudi-client ........................................ SUCCESS [  0.098 s]
	[INFO] hudi-client-common ................................. SUCCESS [  7.816 s]
	[INFO] hudi-hadoop-mr ..................................... SUCCESS [  3.573 s]
	[INFO] hudi-spark-client .................................. SUCCESS [ 17.438 s]
	[INFO] hudi-sync-common ................................... SUCCESS [  0.899 s]
	[INFO] hudi-hive-sync ..................................... SUCCESS [  3.614 s]
	[INFO] hudi-spark-datasource .............................. SUCCESS [  0.063 s]
	[INFO] hudi-spark-common_2.11 ............................. SUCCESS [  8.646 s]
	[INFO] hudi-spark2_2.11 ................................... SUCCESS [ 11.956 s]
	[INFO] hudi-spark_2.11 .................................... SUCCESS [ 35.199 s]
	[INFO] hudi-utilities_2.11 ................................ SUCCESS [  5.049 s]
	[INFO] hudi-utilities-bundle_2.11 ......................... SUCCESS [ 12.905 s]
	[INFO] hudi-cli ........................................... SUCCESS [ 10.460 s]
	[INFO] hudi-java-client ................................... SUCCESS [  2.536 s]
	[INFO] hudi-flink-client .................................. SUCCESS [  6.920 s]
	[INFO] hudi-spark3_2.12 ................................... SUCCESS [  6.546 s]
	[INFO] hudi-dla-sync ...................................... SUCCESS [  1.246 s]
	[INFO] hudi-sync .......................................... SUCCESS [  0.042 s]
	[INFO] hudi-hadoop-mr-bundle .............................. SUCCESS [  4.964 s]
	[INFO] hudi-hive-sync-bundle .............................. SUCCESS [  1.439 s]
	[INFO] hudi-spark-bundle_2.11 ............................. SUCCESS [ 10.566 s]
	[INFO] hudi-presto-bundle ................................. SUCCESS [  5.319 s]
	[INFO] hudi-timeline-server-bundle ........................ SUCCESS [  4.931 s]
	[INFO] hudi-hadoop-docker ................................. SUCCESS [  0.434 s]
	[INFO] hudi-hadoop-base-docker ............................ SUCCESS [  0.016 s]
	[INFO] hudi-hadoop-namenode-docker ........................ SUCCESS [  0.013 s]
	[INFO] hudi-hadoop-datanode-docker ........................ SUCCESS [  0.012 s]
	[INFO] hudi-hadoop-history-docker ......................... SUCCESS [  0.011 s]
	[INFO] hudi-hadoop-hive-docker ............................ SUCCESS [  0.241 s]
	[INFO] hudi-hadoop-sparkbase-docker ....................... SUCCESS [  0.011 s]
	[INFO] hudi-hadoop-sparkmaster-docker ..................... SUCCESS [  0.010 s]
	[INFO] hudi-hadoop-sparkworker-docker ..................... SUCCESS [  0.009 s]
	[INFO] hudi-hadoop-sparkadhoc-docker ...................... SUCCESS [  0.016 s]
	[INFO] hudi-hadoop-presto-docker .......................... SUCCESS [  0.062 s]
	[INFO] hudi-integ-test .................................... SUCCESS [  9.229 s]
	[INFO] hudi-integ-test-bundle ............................. SUCCESS [ 29.610 s]
	[INFO] hudi-examples ...................................... SUCCESS [  5.562 s]
	[INFO] hudi-flink_2.11 .................................... SUCCESS [  3.207 s]
	[INFO] hudi-flink-bundle_2.11 ............................. SUCCESS [ 15.019 s]
	[INFO] ------------------------------------------------------------------------
	
### 编译好之后文件目录对应Hudi下的packaging目录

 
	[xxx@xxx Hudi]# cd packaging/
	[xxx@xxx packaging]# ll
	总用量 36
	drwxr-xr-x 4 root root 4096 9月  16 13:58 hudi-flink-bundle
	drwxr-xr-x 4 root root 4096 9月  16 13:54 hudi-hadoop-mr-bundle
	drwxr-xr-x 4 root root 4096 9月  16 13:54 hudi-hive-sync-bundle
	drwxr-xr-x 4 root root 4096 9月  16 13:56 hudi-integ-test-bundle
	drwxr-xr-x 4 root root 4096 9月  16 13:54 hudi-presto-bundle
	drwxr-xr-x 4 root root 4096 9月  16 13:54 hudi-spark-bundle
	drwxr-xr-x 4 root root 4096 9月  16 13:55 hudi-timeline-server-bundle
	drwxr-xr-x 4 root root 4096 9月  16 13:51 hudi-utilities-bundle

