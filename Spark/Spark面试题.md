# Spark面试题

## Scala的伴生类和伴生对象？

在Scala的类中，与类名相同的对象叫做伴生对象，类和伴生对象之间可以相互访问私有的方法和属性。

例子：

	package com.company.scala.day02
	 
	/**
	  * @description:
	  *              在Scala中，与类名相同的的对象叫做伴生对象
	  *              类和伴生对象之间可以互相访问私有的方法和属性
	  * @author: chunguang.yao
	  * @date: 2019-05-14 23:24
	  */
	class Dog {
	  val id = 1
	  private var name = "zhangsan"
	  def printNam() : Unit = {
	    // 在Dog类中可以方法Dog对象的私有属性
	    println(Dog.CONSTANT + "\t" + name)
	  }
	}
	 
	/**
	  * 伴生对象
	  */
	object  Dog {
	  // 伴生对象中的私有属性
	  private  val  CONSTANT = "Hello"
	  def main(args: Array[String]): Unit = {
	    val d = new Dog
	    // 访问Dog类的私有字段
	    d.name = "lisi"
	    d.printNam()
	  }
	}


伴生对象与伴生类的特点

1、类和它的伴生对象可以互相访问其私有成员

2、单例对象不能new，所以也没有构造参数

3、可以把单例对象当做java中可能会用到的静态方法工具类。

4、作为程序入口的方法必须是静态的，所以main方法必须处在一个单例对象中，而不能写在一个类中。

## Spark和Hadoop的 Shuffle相同和区别？


 Apache Spark 的 Shuffle 过程与 Apache Hadoop 的 Shuffle 过程有着诸多类似，一些概念可直接套用，例如，Shuffle 过程中，提供数据的一端，被称作 Map 端，Map 端每个生成数据的任务称为 Mapper，对应的，接收数据的一端，被称作 Reduce 端，Reduce 端每个拉取数据的任务称为 Reducer，Shuffle 过程本质上都是将 Map 端获得的数据使用分区器进行划分，并将数据发送给对应的 Reducer 的过程。

---

1. 从逻辑角度来讲，Shuffle 过程就是一个 GroupByKey 的过程，两者没有本质区别。只是 MapReduce 为了方便 GroupBy 存在于不同 partition 中的 key/value records，就提前对 key 进行排序。Spark 认为很多应用不需要对 key 排序，就默认没有在 GroupBy 的过程中对 key 排序。

 

2. 从数据流角度讲，两者有差别。MapReduce 只能从一个 Map Stage shuffle 数据，Spark 可以从多个 Map Stages shuffle 数据（这是 DAG 型数据流的优势，可以表达复杂的数据流操作，参见 CoGroup(), join() 等操作的数据流图 SparkInternals/4-shuffleDetails.md at master · JerryLead/SparkInternals · GitHub）。

 

3. Shuffle write/read 实现上有一些区别。以前对 shuffle write/read 的分类是 sort-based 和 hash-based。MapReduce 可以说是 sort-based，shuffle write 和 shuffle read 过程都是基于key sorting 的 (buffering records + in-memory sort + on-disk external sorting)。早期的 Spark 是 hash-based，shuffle write 和 shuffle read 都使用 HashMap-like 的数据结构进行 aggregate (without key sorting)。但目前的 Spark 是两者的结合体，shuffle write 可以是 sort-based (only sort partition id, without key sorting)，shuffle read 阶段可以是 hash-based。因此，目前 sort-based 和 hash-based 已经“你中有我，我中有你”，界限已经不那么清晰。

 

4. 从数据 fetch 与数据计算的重叠粒度来讲，两者有细微区别。MapReduce 是粗粒度，reducer fetch 到的 records 先被放到 shuffle buffer 中休息，当 shuffle buffer 快满时，才对它们进行 combine()。而 Spark 是细粒度，可以即时将 fetch 到的 record 与 HashMap 中相同 key 的 record 进行 aggregate。5. 从性能优化角度来讲，Spark考虑的更全面。MapReduce 的 shuffle 方式单一。Spark 针对不同类型的操作、不同类型的参数，会使用不同的 shuffle write 方式

-----------------------------------------

主要的区别：

一个落盘，一个不落盘，spark就是为了解决mr落盘导致效率低下的问题而产生的，原理还是mr的原理，只是shuffle放在内存中计算了，所以效率提高很多。


## Spark如何划分Stage？

Spark任务会根据RDD之间的依赖关系，形成一个DAG有向无环图，DAG会提交给DAGScheduler，DAGScheduler会把DAG划分相互依赖的多个stage，划分stage的依据就是RDD之间的宽窄依赖。遇到宽依赖就划分stage,每个stage包含一个或多个task任务。然后将这些task以taskSet的形式提交给TaskScheduler运行。    

stage是由一组并行的task组成。


## spark的有几种部署模式，每种模式特点？

### 本地模式
Spark不一定非要跑在hadoop集群，可以在本地，起多个线程的方式来指定。方便调试，本地模式分三类

local：只启动一个executor

local[k]: 启动k个executor

local：启动跟cpu数目相同的 executor

### standalone模式

分布式部署集群，自带完整的服务，资源管理和任务监控是Spark自己监控，这个模式也是其他模式的基础

### Spark on yarn模式

分布式部署集群，资源和任务监控交给yarn管理

粗粒度资源分配方式，包含cluster和client运行模式

cluster 适合生产，driver运行在集群子节点，具有容错功能

client 适合调试，dirver运行在客户端

###Spark On Mesos模式


## Spark技术栈有哪些组件，每个组件都有什么功能，适合什么应用场景？


### Spark core

是其它组件的基础，spark的内核

主要包含：有向循环图、RDD、Lingage、Cache、broadcast等


### SparkStreaming

是一个对实时数据流进行高通量、容错处理的流式处理系统

将流式计算分解成一系列短小的批处理作业

### Spark sql：

能够统一处理关系表和RDD，使得开发人员可以轻松地使用SQL命令进行外部查询

### MLBase
是Spark生态圈的一部分专注于机器学习，让机器学习的门槛更低

MLBase分为四部分：MLlib、MLI、ML Optimizer和MLRuntime。

### GraphX
是Spark中用于图和图并行计算


## spark有哪些角色

master：管理集群和节点，不参与计算。

worker：计算节点，进程本身不参与计算，和master汇报。

Driver：运行程序的main方法，创建spark context对象。

spark context：控制整个application的生命周期，包括dagsheduler和task scheduler等组件。

client：用户提交程序的入口


## spark工作机制

用户在client端提交作业后，会由Driver运行main方法并创建spark context上下文。

执行add算子，形成dag图输入dagscheduler

按照add之间的依赖关系划分stage输入task scheduler

task scheduler会将stage划分为taskset分发到各个节点的executor中执行

## Spark应用程序的执行过程

构建Spark Application的运行环境（启动SparkContext）

SparkContext向资源管理器（可以是Standalone、Mesos或YARN）注册并申请运行Executor资源；

资源管理器分配Executor资源，Executor运行情况将随着心跳发送到资源管理器上；

SparkContext构建成DAG图，将DAG图分解成Stage，并把Taskset发送给Task Scheduler

Executor向SparkContext申请Task，Task Scheduler将Task发放给Executor运行，
SparkContext将应用程序代码发放给Executor。

Task在Executor上运行，运行完毕释放所有资源。


## Hadoop和Spark联系与区别

Hadoop实质上更多是一个分布式数据基础设施: 它将巨大的数据集分派到一个由普通计算机组成的集群中的多个节点进行存储，意味着您不需要购买和维护昂贵的服务器硬件。同时，Hadoop还会索引和跟踪这些数据，让大数据处理和分析效率达到前所未有的高度。

Spark，则是那么一个专门用来对那些分布式存储的大数据进行处理的工具，它并不会进行分布式数据的存储。

Spark数据处理速度秒杀MapReduce

Spark因为其处理数据的方式不一样，会比MapReduce快上很多。MapReduce是分步对数据进行处理的: 从集群中读取数据，进行一次处理，将结果写到集群，从集群中读取更新后的数据，进行下一次的处理，将结果写到集群，等等…  Spark是采用了流式处理的方法。速度很快很多。


## Spark和Hadoop的灾难异同

者的灾难恢复方式迥异，但是都很不错。因为Hadoop将每次处理后的数据都写入到磁盘上，所以其天生就能很有弹性的对系统错误进行处理。

Spark的数据对象存储在分布于数据集群中的叫做弹性分布式数据集(RDD: Resilient Distributed Dataset)中。“这些数据对象既可以放在内存，也可以放在磁盘，所以RDD同样也可以提供完成的灾难恢复功能。


## 什么是宽依赖，什么是窄依赖？哪些算子是宽依赖，哪些是窄依赖？

窄依赖就是一个父RDD分区对应一个子RDD分区，如map，filter或者多个父RDD分区对应一个子RDD分区，如co-partioned join

宽依赖是一个父RDD分区对应非全部的子RDD分区，如groupByKey，ruduceByKey或者一个父RDD分区对应全部的子RDD分区，如未经协同划分的join


