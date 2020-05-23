# Spark 之线性回归


## 准备数据

商品价格与消费者输入之间的关系

### 准备目录

	mkdir -p /root/spark_demo/linear_regress
	mkdir /root/spark_demo/linear_regress/input /root/spark_demo/linear_regress/output /root/spark_demo/linear_regress/model

### 构建样本数据

	vi /root/spark_demo/linear_regress/input/lr.data

插入

	5|1,1
	8|1,2
	7|2,1
	13|2,3
	18|3,4



格式为：

标签,特征值1 特征值2 特征值3...


## 导入Hadoop

	hadoop dfs -mkdir /spark_data
	hadoop dfs -put /root/spark_demo/linear_regress/input/lr.data /spark_data/lr.data

	#查看
	hadoop dfs -ls /spark_data
	hadoop dfs -cat /spark_data/lr.data



## 源码

新建maven工程


## 部署


### 转到spark目录
cd /root/spark-2.4.4-bin-hadoop2.7/bin

### 使用spark submit提交任务
./spark-submit --class com.wzy.LinearRegression --master spark://Master001:7077 --executor-memory 512M --total-executor-cores 2 /root/spark_linear_regresion-1.0-SNAPSHOT.jar 

