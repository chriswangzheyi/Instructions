# Apache Doris 环境安装部署

参考：https://zhuanlan.zhihu.com/p/411093925

## 安装步骤

### 创建镜像

	docker pull apache/incubator-doris:build-env-1.3.1
  
 ### 源码下载编译
 
 	https://github.com/baidu/palo/archive/refs/tags/PALO-0.14.13-release.tar.gz

 放置到
 
 	/Users/zheyiwang/Documents/doris/palo-PALO-0.14.13-release
 
 ### 启动镜像
 
 	docker run -it --name doris-build-1.3.1 -v /Users/zheyiwang/Documents/doris/.m2:/root/.m2 -v /Users/zheyiwang/Documents/doris/:/root/doris/ apache/incubator-doris:build-env-1.3.1
 	
 此时会进入容器内，进入到你的doris源码目录
 
 	cd /root/doris/doris-0.14.13
 	sh build.sh
 	
 成功后显示：
 
	 [INFO] doris-fe-common 1.0.0 .............................. SUCCESS [01:00 min]
	[INFO] spark-dpp 1.0.0 .................................... SUCCESS [ 31.844 s]
	[INFO] fe-core 3.4.0 ...................................... SUCCESS [02:25 min]
	[INFO] ------------------------------------------------------------------------
	[INFO] BUILD SUCCESS
	[INFO] ------------------------------------------------------------------------
	[INFO] Total time:  03:58 min
	[INFO] Finished at: 2022-09-25T01:42:59Z
	[INFO] ------------------------------------------------------------------------
	***************************************
	Successfully build Doris
	***************************************

编译好的安装包在源码根目录：output目录下，拷贝出来就是可以安装

### 编译Doris Broker	

	cd /root/doris/palo-PALO-0.14.13-release/fs_brokers/apache_hdfs_broker
	sh build.sh 
	
### Doris 安装

#### 注意事项

* FE 的磁盘空间主要用于存储元数据，包括日志和 image。通常从几百 MB 到几个 GB 不等。
* BE 的磁盘空间主要用于存放用户数据，总磁盘空间按用户总数据量 * 3（3副本）计算，然后再预留额外 40% 的空间用作后台 compaction 以及一些中间数据的存放。
* 一台机器上可以部署多个 BE 实例，但是只能部署一个 FE。如果需要 3 副本数据，那么至少需要 3 台机器各部署一个 BE 实例（而不是1台机器部署3个BE实例）。多个FE所在服务器的时钟必须保持一致（允许最多5秒的时钟偏差）


#### 关于FE节点数量

* FE 角色分为 Follower 和 Observer，（Leader 为 Follower 组中选举出来的一种角色，以下统称 Follower）。
* FE 节点数据至少为1（1 个 Follower）。当部署 1 个 Follower 和 1 个 Observer 时，可以实现读高可用。当部署 3 个 Follower 时，可以实现读写高可用（HA）。
* Follower 的数量必须为奇数，Observer 数量随意。
* 根据以往经验，当集群可用性要求很高时（比如提供在线业务），可以部署 3 个 Follower 和 1-3 个 Observer。如果是离线业务，建议部署 1 个 Follower 和 1-3 个 Observer。


#### 部署过程

* 通常我们建议 10 ~ 100 台左右的机器，来充分发挥 Doris 的性能（其中 3 台部署 FE（HA），剩余的部署 BE
* 当然，Doris的性能与节点数量及配置正相关。在最少4台机器（一台 FE，三台 BE，其中一台 BE 混部一个 Observer FE 提供元数据备份），以及较低配置的情况下，依然可以平稳的运行 Doris。
* 如果 FE 和 BE 混部，需注意资源竞争问题，并保证元数据目录和数据目录分属不同磁盘。
* 这里我们使用3个FE，5个BE节点，来搭建一个完整的支持高可用的Doris集群,部署角色如下

（未完待续）