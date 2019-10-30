#HDFS 操作

###列出文件目录

	hdfs dfs -ls 路径

### 在HDFS创建文件夹

	hdfs dfs -mkdir 文件夹名称

### 上传文件至HDFS

	hdfs dfs -put 源路径 目标存放路径

### 从HDFS上下载文件

	hdfs dfs -get HDFS文件路径 本地存放路径

### 查看HDFS上某个文件的内容

	hdfs dfs -text(或cat) HDFS上的文件存放路径

### 统计目录下各文件的大小

	hdfs dfs -du 目录路径

### 删除HDFS上某个文件或者文件夹

	hdfs dfs -rm 文件存放文件
	hdfs dfs -rm -r 文件存放文件