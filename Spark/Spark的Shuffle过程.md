# Spark的Shuffle过程

##Shuffle Write：
一批任务（ShuffleMapTask）将程序输出的临时数据并写到本地磁盘。由于每个任务产生的数据要被下一个阶段的每个任务读取一部分，因此存入磁盘时需对数据分区，分区可以使用Hash与Sort两种方法；

##Shuffle Read：
下一个阶段启动一批新任务（ResultTask），它们各自启动一些线程远程读取Shuffle Write产生的数据；

##Aggregate：
一旦数据被远程拷贝过来后，接下来需按照key将数据组织在一起，为后续计算做准备。
