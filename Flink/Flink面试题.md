# Flink面试题

## Flink 如何处理反压？

### 反压出现的场景
**数据的消费速度小于数据的生产速度。**
反压经常出现在促销、热门活动等场景。短时间内流量陡增造成数据的堆积或者消费速度变慢。

### 反压监控方法

**通过Flink Web UI发现反压问题**
Flink 的 TaskManager 会每隔 50 ms 触发一次反压状态监测，共监测 100 次，并将计算结果反馈给 JobManager，最后由 JobManager 进行计算反压的比例，然后进行展示。

### 反压问题定位和处理

Flink会因为数据堆积和处理速度变慢导致checkpoint超时，而checkpoint是Flink保证数据一致性的关键所在，最终会导致数据的不一致发生。

* 数据倾斜：可以在 Flink 的后台管理页面看到每个 Task 处理数据的大小。当数据倾斜出现时，通常是简单地使用类似 KeyBy 等分组聚合函数导致的，需要用户将热点 Key 进行预处理，降低或者消除热点 Key 的影

* GC：不合理的设置 TaskManager 的垃圾回收参数会导致严重的 GC 问题，我们可以通过 -XX:+PrintGCDetails 参数查看 GC 的日志。

* 代码本身：开发者错误地使用 Flink 算子，没有深入了解算子的实现机制导致性能问题。我们可以通过查看运行机器节点的 CPU 和内存情况定位问题。

## 如何处理生产环境中的数据倾斜问题？

业务上要尽量避免热点 key 的设计，例如我们可以把北京、上海等热点城市分成不同的区域，并进行单独处理；

技术上出现热点时，要调整方案打散原来的 key，避免直接聚合；此外 Flink 还提供了大量的功能可以避免数据倾斜。

### Flink 任务数据倾斜场景和解决方案

#### 两阶段聚合解决 KeyBy 热点：

首先把分组的 key 打散，比如加随机后缀；
对打散后的数据进行聚合；
把打散的 key 还原为真正的 key；

#### GroupBy + Aggregation 分组聚合热点问题:

将SQL 拆成了内外两层，第一层通过随机打散 100 份的方式减少数据热点

#### Flink 消费 Kafka 上下游并行度不一致导致的数据倾斜


## flink海量数据高效去重

* 基于布隆过滤器（BloomFilter）
* 基于BitMap
* 基于外部数据库


## Flink 任务出现很高的延迟，你会如何入手解决类似问题

在 Flink 的后台任务管理中，可以看到 Flink 的哪个算子和 task 出现了反压；

资源调优和算子调优：资源调优即对作业中的 Operator 并发数（Parallelism）、CPU（Core）、堆内存（Heap_memory）等参数进行调优；

作业参数调优：并行度的设置、State 的设置、Checkpoint 的设置。


## exactly-once 的保证

下级存储支持事务：Flink可以通过实现两阶段提交和状态保存来实现端到端的一致性语义。 分为以下几个步骤：

* 1）开始事务（beginTransaction）创建一个临时文件夹，来写把数据写入到这个文件夹里面
* 2）预提交（preCommit）将内存中缓存的数据写入文件并关闭
* 3）正式提交（commit）将之前写完的临时文件放入目标目录下。这代表着最终的数据会有一些延迟
* 4）丢弃（abort）丢弃临时文件
* 5）若失败发生在预提交成功后，正式提交前。可以根据状态来提交预提交的数据，也可删除预提交的数据。

下级存储不支持事务：
具体实现是幂等写入，需要下级存储具有幂等性写入特性。


## checkpoint 与 spark 比较

spark streaming 的 checkpoint 仅仅是针对 driver 的故障恢复做了数据和元数据的checkpoint。而 flink 的 checkpoint 机制要复杂了很多，它采用的是轻量级的分布式快照，实现了每个算子的快照，及流动中的数据的快照。

可以是内存，文件系统，或者 RocksDB。

## 三种时间语义

* Event Time：这是实际应用最常见的时间语义，指的是事件创建的时间，往往跟watermark结合使用
* Processing Time：指每一个执行基于时间操作的算子的本地系统时间，与机器相关。适用场景：没有事件时间的情况下，或者对实时性要求超高的情况
* Ingestion Time：指数据进入Flink的时间。适用场景：存在多个 Source Operator 的情况下，每个 Source Operator 可以使用自己本地系统时钟指派 Ingestion Time。后续基于时间相关的各种操作， 都会使用数据记录中的 Ingestion Time


<<<<<<< Updated upstream
## Flink CEP 编程中当状态没有到达的时候会将数据保存在哪里？

在流式处理中，CEP 当然是要支持 EventTime 的，那么相对应的也要支持数据的迟到现象，也就是 watermark 的处理逻辑。CEP 对未匹配成功的事件序列的处理，和迟到数据是类似的。在 Flink CEP 的处理逻辑中，状态没有满足的和迟到的数据，都会存储在一个 Map 数据结构中，也就是说，如果我们限定判断事件序列的时长为 5 分钟，那么内存中就会存储 5 分钟的数据，这在我看来，也是对内存的极大损伤之一。

## Flink有没有重启策略？说说有哪几种？

* 固定延迟重启策略（Fixed Delay Restart Strategy）
* 故障率重启策略（Failure Rate Restart Strategy）
* 没有重启策略（No Restart Strategy）
* Fallback重启策略（Fallback Restart Strategy）


## 说说Flink中的状态存储？

MemoryStateBackend、FsStateBackend、RocksDBStateBackend。


## Flink的内存管理是如何做的?

link 并不是将大量对象存在堆上，而是将对象都序列化到一个预分配的内存块上。此外，Flink大量的使用了堆外内存。如果需要处理的数据超出了内存限制，则会将部分数据存储到硬盘上。Flink 为了直接操作二进制数据实现了自己的序列化框架。

理论上Flink的内存管理分为三部分：

* Network Buffers：这个是在TaskManager启动的时候分配的，这是一组用于缓存网络数据的内存，每个块是32K，默认分配2048个，可以通过“taskmanager.network.numberOfBuffers”修改
* Memory Manage pool：大量的Memory Segment块，用于运行时的算法（Sort/Join/Shuffle等），这部分启动的时候就会分配。下面这段代码，根据配置文件中的各种参数来计算内存的分配方法。（heap or off-heap，这个放到下节谈），内存的分配支持预分配和lazy load，默认懒加载的方式。
* User Code，这部分是除了Memory Manager之外的内存用于User code和TaskManager本身的数据结构。
=======
## Flink集群有哪些角色？各自有什么作用？

Flink程序在运行时主要有TaskManager，JobManager，Client三种角色。

JobManager扮演着集群中的管理者Master的角色，它是整个集群的协调者，负责接收Flink Job，协调检查点，Failover 故障恢复等，同时管理Flink集群中从节点TaskManager。

TaskManager是实际负责执行计算的Worker，在其上执行Flink Job的一组Task，每个TaskManager负责管理其所在节点上的资源信息，如内存、磁盘、网络，在启动的时候将资源的状态向JobManager汇报。

Client是Flink程序提交的客户端，当用户提交一个Flink程序时，会首先创建一个Client，该Client首先会对用户提交的Flink程序进行预处理，并提交到Flink集群中处理，所以Client需要从用户提交的Flink程序配置中获取JobManager的地址，并建立到JobManager的连接，将Flink Job提交给JobManager。

## 么提交的实时任务，有多少Job Manager？

1）我们使用yarn session模式提交任务；另一种方式是每次提交都会创建一个新的Flink 集群，为每一个job提供资源，任务之间互相独立，互不影响，方便管理。任务执行完成之后创建的集群也会消失。线上命令脚本如下：

	bin/yarn-session.sh -n 7 -s 8 -jm 3072 -tm 32768 -qu root.. -nm - -d

其中申请7个 taskManager，每个 8 核，每个 taskmanager 有 32768M 内存。

2）集群默认只有一个 Job Manager。但为了防止单点故障，我们配置了高可用。对于standlone模式，我们公司一般配置一个主 Job Manager，两个备用 Job Manager，然后结合 ZooKeeper 的使用，来达到高可用；对于yarn模式，yarn在Job Mananger故障会自动进行重启，所以只需要一个，我们配置的最大重启次数是10次。


## 说说Flink中的窗口

Flink 支持两种划分窗口的方式，按照time和count。

如果根据时间划分窗口，那么它就是一个time-window 如果根据数据划分窗口，那么它就是一个count-window。flink支持窗口的两个重要属性（size和interval）如果size=interval,那么就会形成tumbling-window(无重叠数据) 如果size>interval,那么就会形成sliding-window(有重叠数据) 如果size< interval, 那么这种窗口将会丢失数据。比如每5秒钟，统计过去3秒的通过路口汽车的数据，将会漏掉2秒钟的数据。通过组合可以得出四种基本窗口：

* time-tumbling-window 无重叠数据的时间窗口，设置方式举例：timeWindow(Time.seconds(5))
* time-sliding-window有重叠数据的时间窗口，设置方式举例：timeWindow(Time.seconds(5), Time.seconds(3))
* count-tumbling-window无重叠数据的数量窗口，设置方式举例：countWindow(5)
* count-sliding-window 有重叠数据的数量窗口，设置方式举例：countWindow(5,3)

## 说一下Flink状态机制

Flink在做计算的过程中经常需要存储中间状态，来避免数据丢失和状态恢复。选择的状态存储策略不同，会影响状态持久化如何和 checkpoint 交互。

Flink提供了三种状态存储方式：MemoryStateBackend、FsStateBackend、RocksDBStateBackend。

## Flink分布式快照的原理是什么

Flink的容错机制的核心部分是制作分布式数据流和操作算子状态的一致性快照。 这些快照充当一致性checkpoint，系统可以在发生故障时回滚。 Flink用于制作这些快照的机制在“分布式数据流的轻量级异步快照”中进行了描述。 它受到分布式快照的标准Chandy-Lamport算法的启发，专门针对Flink的执行模型而定制。

barriers在数据流源处被注入并行数据流中。快照n的barriers被插入的位置（我们称之为Sn）是快照所包含的数据在数据源中最大位置。

例如，在Apache Kafka中，此位置将是分区中最后一条记录的偏移量。 将该位置Sn报告给checkpoint协调器（Flink的JobManager）。

然后barriers向下游流动。当一个中间操作算子从其所有输入流中收到快照n的barriers时，它会为快照n发出barriers进入其所有输出流中。

一旦sink操作算子（流式DAG的末端）从其所有输入流接收到barriers n，它就向checkpoint协调器确认快照n完成。
在所有sink确认快照后，意味快照着已完成。一旦完成快照n，job将永远不再向数据源请求Sn之前的记录，因为此时这些记录（及其后续记录）将已经通过整个数据流拓扑，也即是已经被处理结束。

## 介绍一下Flink的CEP机制

CEP全称为Complex Event Processing，复杂事件处理
Flink CEP是在 Flink 中实现的复杂事件处理（CEP）库
CEP 允许在无休止的事件流中检测事件模式，让我们有机会掌握数据中重要的部分
一个或多个由简单事件构成的事件流通过一定的规则匹配，然后输出用户想得到的数据 —— 满足规则的复杂事件


## Flink CEP 编程中当状态没有到达的时候会将数据保存在哪里？

在流式处理中，CEP 当然是要支持 EventTime 的，那么相对应的也要支持数据的迟到现象，也就是watermark的处理逻辑。CEP对未匹配成功的事件序列的处理，和迟到数据是类似的。在 Flink CEP的处理逻辑中，状态没有满足的和迟到的数据，都会存储在一个Map数据结构中，也就是说，如果我们限定判断事件序列的时长为5分钟，那么内存中就会存储5分钟的数据，这在我看来，也是对内存的极大损伤之一。


## Flink 在使用 Window 时出现数据倾斜，你有什么解决办法？

这里 window 产生的数据倾斜指的是不同的窗口内积攒的数据量不同，主要是由源头 数据的产生速度导致的差异。核心思路：1.重新设计 key 2.在窗口计算前做预聚合

## Flink 的内存管理是如何做的

Flink 并不是将大量对象存在堆上，而是将对象都序列化到一个预分配的内存块 上。此外，Flink 大量的使用了堆外内存。如果需要处理的数据超出了内存限制， 则会将部分数据存储到硬盘上。Flink 为了直接操作二进制数据实现了自己的序 列化框架。

## Flink 是如何支持批流一体的

Flink 的开发者认为批处理是流处理的一种特殊情况。 批处理是有限的流处理。Flink 使用一个引擎支持了 DataSet API 和 DataStream API。

## Checkpoint机制

主要是当Flink开启Checkpoint的时候，会往Source端插入一条barrir，然后这个barrir随着数据流向一直流动，当流入到一个算子的时候，这个算子就开始制作checkpoint，制作的是从barrir来到之前的时候当前算子的状态，将状态写入状态后端当中。然后将barrir往下流动，当流动到keyby 或者shuffle算子的时候，例如当一个算子的数据，依赖于多个流的时候，这个时候会有barrir对齐，也就是当所有的barrir都来到这个算子的时候进行制作checkpoint，依次进行流动，当流动到sink算子的时候，并且sink算子也制作完成checkpoint会向jobmanager 报告 checkpoint n 制作完成。


## 二阶段提交机制

Flink 提供了CheckpointedFunction与CheckpointListener这样两个接口，CheckpointedFunction中有snapshotState方法，每次checkpoint触发执行方法，通常会将缓存数据放入状态中，可以理解为一个hook，这个方法里面可以实现预提交，CheckpointListyener中有notifyCheckpointComplete方法，checkpoint完成之后的通知方法，这里可以做一些额外的操作。例如FLinkKafkaConumerBase使用这个来完成Kafka offset的提交，在这个方法里面可以实现提交操作。在2PC中提到如果对应流程例如某个checkpoint失败的话，那么checkpoint就会回滚，不会影响数据一致性，那么如果在通知checkpoint成功的之后失败了，那么就会在initalizeSate方法中完成事务的提交，这样可以保证数据的一致性。最主要是根据checkpoint的状态文件来判断的。

## flink和spark区别

flink是一个类似spark的“开源技术栈”，因为它也提供了批处理，流式计算，图计算，交互式查询，机器学习等。flink也是内存计算，比较类似spark，但是不一样的是，spark的计算模型基于RDD，将流式计算看成是特殊的批处理，他的DStream其实还是RDD。而flink吧批处理当成是特殊的流式计算，但是批处理和流式计算的层的引擎是两个，抽象了DataSet和DataStream。flink在性能上也表现的很好，流式计算延迟比spark少，能做到真正的流式计算，而spark只能是准流式计算。而且在批处理上，当迭代次数变多，flink的速度比spark还要快，所以如果flink早一点出来，或许比现在的Spark更火。

## Flink window join

1、window join，即按照指定的字段和滚动滑动窗口和会话窗口进行 inner join
2、是coGoup 其实就是left join 和 right join，
3、interval join 也就是 在窗口中进行join 有一些问题，因为有些数据是真的会后到的，时间还很长，那么这个时候就有了interval join但是必须要是事件时间，并且还要指定watermark和水位以及获取事件时间戳。并且要设置 偏移区间，因为join 也不能一直等的。

## flink窗口函数有哪些

* Tumbing window
* Silding window
* Session window
* Count winodw

## keyedProcessFunction 是如何工作的。假如是event time的话

yedProcessFunction 是有一个ontime 操作的，假如是 event时间的时候 那么 调用的时间就是查看，event的watermark 是否大于 trigger time 的时间，如果大于则进行计算，不大于就等着，如果是kafka的话，那么默认是分区键最小的时间来进行触发。


## flink 维表关联怎么做的

* 1、async io
* 2、broadcast
* 3、async io + cache
* 4、open方法中读取，然后定时线程刷新，缓存更新是先删除，之后再来一条之后再负责写入缓存


## Flink checkpoint的超时问题 如何解决

* 1、是否网络问题
* 2、是否是barrir问题
* 3、查看webui，是否有数据倾斜
* 4、有数据倾斜的话，那么解决数据倾斜后，会有改善，


## Flink 监控你们怎么做的

* 1、我们监控了Flink的任务是否停止
* 2、我们监控了Flink的Kafka的LAG
* 3、我们会进行实时数据对账，例如销售额。


##Flink 有数据丢失的可能吗

Flink有三种数据消费语义：

* At Most Once 最多消费一次 发生故障有可能丢失
* At Least Once 最少一次 发生故障有可能重复
* Exactly-Once 精确一次 如果产生故障，也能保证数据不丢失不重复。

>>>>>>> Stashed changes
