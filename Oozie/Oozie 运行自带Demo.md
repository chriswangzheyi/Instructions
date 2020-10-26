# Oozie 运行自带Demo

## 前置步骤


### 启动Hadoop的historyserver

	sbin/mr-jobhistory-daemon.sh start historyserver
	
	
### 解压example文件和share文件

	cd /root/bigdata/oozie-5.1.0
	tar -zxvf oozie-examples.tar.gz
	tar -zxvf oozie-sharelib-5.1.0.tar.gz
	
## 修改配置文件

### job.properties 

	cd /root/bigdata/oozie-5.1.0/examples/apps/map-reduce
	vi job.properties 
	
修改参数（namenode和resourcw）：
	
	nameNode=hdfs://172.18.156.87:9000
	resourceManager=http://172.18.156.87:8032
	queueName=default
	examplesRoot=examples
	
	oozie.wf.application.path=${nameNode}/user/${user.name}/${examplesRoot}/apps/map-reduce/workflow.xml
	outputDir=map-reduce	
	
注意：nameNode和resourceManager端口配错了可能会导致任务启动不了或者一直Runing
	
### workflow.xml

在该文件下可以定义工作流


## 部署测试项目

	hdfs dfs -mkdir -p /user/root/
	hdfs dfs -put examples/ /user/root/
	hdfs dfs -put share/ /user/root/
	
	oozie job -oozie http://172.18.156.87:11000/oozie/ -config /root/bigdata/oozie-5.1.0/examples/apps/map-reduce/job.properties -run
	
	
输出：
	
	[root@wangzheyi map-reduce]# oozie job -oozie http://172.18.156.87:11000/oozie/ -config /root/bigdata/oozie-5.1.0/examples/apps/map-reduce/job.properties -run
	job: 0000000-201017093429032-oozie-root-W
	
	
## 检查状态

	[root@wangzheyi ~]# oozie admin -oozie http://localhost:11000/oozie -status
	System mode: NORMAL
	
## 任务成功界面

访问：

	http://47.112.142.231:11000/oozie/

![](Images/2.png)

## 查看Demo的运行结果

	hdfs dfs -cat /user/root/examples/output-data/map-reduce/part-00000

输出：

	[root@wangzheyi ~]# hdfs dfs -cat /user/root/examples/output-data/map-reduce/part-00000
	0       To be or not to be, that is the question;
	42      Whether 'tis nobler in the mind to suffer
	84      The slings and arrows of outrageous fortune,
	129     Or to take arms against a sea of troubles,
	172     And by opposing, end them. To die, to sleep;
	217     No more; and by a sleep to say we end
	255     The heart-ache and the thousand natural shocks
	302     That flesh is heir to ? 'tis a consummation
	346     Devoutly to be wish'd. To die, to sleep;
	387     To sleep, perchance to dream. Ay, there's the rub,
	438     For in that sleep of death what dreams may come,
	487     When we have shuffled off this mortal coil,
	531     Must give us pause. There's the respect
	571     That makes calamity of so long life,
	608     For who would bear the whips and scorns of time,
	657     Th'oppressor's wrong, the proud man's contumely,
	706     The pangs of despised love, the law's delay,
	751     The insolence of office, and the spurns
	791     That patient merit of th'unworthy takes,
	832     When he himself might his quietus make
	871     With a bare bodkin? who would fardels bear,
	915     To grunt and sweat under a weary life,
	954     But that the dread of something after death,
	999     The undiscovered country from whose bourn
	1041    No traveller returns, puzzles the will,
	1081    And makes us rather bear those ills we have
	1125    Than fly to others that we know not of?
	1165    Thus conscience does make cowards of us all,
	1210    And thus the native hue of resolution
	1248    Is sicklied o'er with the pale cast of thought,
	1296    And enterprises of great pitch and moment
	1338    With this regard their currents turn awry,
	1381    And lose the name of action.
	
可以看到内容按key的顺序被排序了