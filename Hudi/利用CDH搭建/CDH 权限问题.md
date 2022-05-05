# CDH 权限问题


CDH环境下Hadoop平台最高权限用户是hdfs，属于supergroup组。默认HDFS会开启权限认证，所以操作时，需要将root用户切换到hdfs用户，否则会报错。

###1、创建用户（所有节点）
	useradd test

###2、创建用户组 （所有节点）
	groupadd supergroup

###3、将用户添加到用户组中（所有节点）
	usermod -a -G supergroup test

###4、同步用户组到hdfs文件系统用户组
	su - hdfs -s /bin/bash -c "hdfs dfsadmin -refreshUserToGroupsMappings"