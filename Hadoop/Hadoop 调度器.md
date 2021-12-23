# Hadoop 调度器

Hadoop 作业调度主要有三种：FIFO、Capacity Scheduler 和 Fari Scheduler。 Hadoop 3.1.3 默认的资源调度器是Capacity Scheduler。

## 先进先出调度器（FIFO）
![](Images/2.png)

## 容器调度器 （Capacity Scheduler）
![](Images/3.png)

支持多个队列，每个队列可配置一定的资源量，每个队列采用FIFO调度策略。


## 公平调度器 （Fair Scheduler）

![](Images/4.png)

支持多个队列，每个队列中的资源量可以配置，同一个队列中的作业公平共享队列中的所有资源。在资源有限的情况下，每个job理想情况下获得的计算资源与实际获得的计算资源存在一种差距，这个差距就叫做缺额。在同一个队列中，job的资源缺额越大，越先获得资源优先执行。