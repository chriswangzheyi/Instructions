# Zookeeper基本命令

	help	显示所有操作命令
	ls path [watch]	使用 ls 命令来查看当前znode中所包含的内容
	ls2 path [watch]	查看当前节点数据并能看到更新次数等数据
	create	普通创建
	create	-s 含有序列
	create	-e 临时（重启或者超时消失）
	get path [watch]	获得节点的值
	set	设置节点的具体值
	stat	查看节点状态
	delete	删除节点
	deleteall	递归删除节点
