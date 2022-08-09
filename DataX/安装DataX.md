# 安装DataX

## 环境

Mac电脑

python2.7


## 步骤

下载

	http://datax-opensource.oss-cn-hangzhou.aliyuncs.com/datax.tar.gz
	
解压后

	$ cd  {YOUR_DATAX_HOME}/bin
	$ python datax.py {YOUR_JOB.json}
	
自检脚本：    

 	python {YOUR_DATAX_HOME}/bin/datax.py {YOUR_DATAX_HOME}/job/job.json
 	例子：
 	(python27) zheyiwang@ZHEYIdeMacBook-Pro bin % python /Users/zheyiwang/Downloads/datax/bin/datax.py /Users/zheyiwang/Downloads/datax/job/job.json 

输出：

......

	2022-08-09 15:47:35.127 [job-0] INFO  JobContainer - PerfTrace not enable!
	2022-08-09 15:47:35.128 [job-0] INFO  StandAloneJobContainerCommunicator - Total 100000 records, 2600000 bytes | Speed 253.91KB/s, 10000 records/s | Error 0 records, 0 bytes |  All Task WaitWriterTime 0.014s |  All Task WaitReaderTime 0.022s | Percentage 100.00%
	2022-08-09 15:47:35.129 [job-0] INFO  JobContainer - 
	任务启动时刻                    : 2022-08-09 15:47:25
	任务结束时刻                    : 2022-08-09 15:47:35
	任务总计耗时                    :                 10s
	任务平均流量                    :          253.91KB/s
	记录写入速度                    :          10000rec/s
	读出记录总数                    :              100000
	读写失败总数                    :                   0

## 错误处理

mysql-connector版本需要匹配，

	mysql-connector-java-8.0.17.jar

放入lib目录下	