# Hadoop 合并小文件
参考：https://www.it1352.com/1595924.html

##  步骤


### 第1步:创建一个tmp目录

	hdfs dfs -mkdir tmp_files	
	
### 第2步:在某个时间点将所有小文件移至tmp目录

	hdfs dfs -mkdir /tmp_files
	hdfs dfs -mv /input/*.txt /tmp_files
	
### 第3步：在hadoop-streaming jar的帮助下合并小文件
	                   
	hadoop jar /opt/cloudera/parcels/CDH/lib/hadoop-mapreduce/hadoop-streaming.jar \
	                   -Dmapred.reduce.tasks=1 \
	                   -input /tmp_files \
	                   -output /output \
	                   -mapper cat \
	                   -reducer cat
	                   
### 第4步：将输出移至输入文件夹

	hdfs dfs -mv /output/part-00000 /input/large_file.txt
	
### 第5步：删除输出

	 hdfs dfs -rm -r /output
	 
### 第6步：从tmp中删除所有文件

	hdfs dfs -rm /tmp_files/*.txt