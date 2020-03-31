# Kafka中的选举

## 控制器的选举

控制器组件（Controller）,是Apache kafka的核心组件。它的主要作用是在Apache Zookeeper的帮助下管理和协调整个kafka集群（社区正在计划去掉zookeeper的依赖）。集群中的任意一台Broker都可以成为控制器，在一个正常运行的集群中目前只能有一个控制器。

实际上，在Broker启动的时候，会尝试去zookeeper中创建/controller节点。第一个成功创建/controller的节点Broker会被指定为控制器。
控制器的作用：

- **主题管理：**主题的创建、删除、增加分区等。当我们在任意一台Broker中执行kafka-topic脚本时，它们会自动找到控制器，并把工作交给控制器来执行。
- **分区重分配：** 分区重分配主要是指，kafka-reassign-partition脚本提供的对已有主题进行细粒度的分配功能。这部分功能也是控制器来实现。
- **Preferred(优先)领导者选举**：Preferred领导者选举主要是kafka为了避免部分Broker负载过重而提供的一种换Leader的方案。
- **集群成员管理：**新增Broker、Broker主动关闭、Broker宕机
- **数据服务：** 就是想其他Broker提供数据服务。控制器上保存了最全的集群元数据信息，其他所有Broker会定期接受控制器发来的元数据更新请求，从而更新其内存中的缓存数据。
下面用一张图来说明一下控制器中保存了什么信息：

![](../Images/8.png)


故障转移指的是，当运⾏中的控制器突然宕机或意外终⽌时，Kafka 能够快速地感知到，并⽴即 启⽤备⽤控制器来代替之前失败的控制器。这个过程就被称为 Failover，该过程是⾃动完成的， ⽆需你⼿动⼲预。 接下来，我们⼀起来看⼀张图，它简单地展示了控制器故障转移的过程。

![](../Images/9.png)

最开始时，Broker 0 是控制器。当 Broker 0 宕机后，ZooKeeper 通过 Watch 机制感知到并删 除了 /controller 临时节点。之后，所有存活的 Broker 开始竞选新的控制器身份。Broker 3 最 终赢得了选举，成功地在 ZooKeeper上重建了/controller节点。之后，Broker3会从ZooKeeper 中读取集群元数据信息，并初始化到⾃⼰的缓存中. ⾄此,控制器的Failover完成，可以⾏使正常的⼯作职责了。

**Kafka Controller的选举是依赖Zookeeper来实现的，**


## 分区副本选举机制

在kafka的集群中，会存在着多个主题topic，在每一个topic中，又被划分为多个partition，为了防止数据不丢失，每一个partition又有多个副本，在整个集群中，总共有三种副本角色：

首领副本（leader）：也就是leader主副本，每个分区都有一个首领副本，为了保证数据一致性，所有的生产者与消费者的请求都会经过该副本来处理。

跟随者副本（follower）：除了首领副本外的其他所有副本都是跟随者副本，跟随者副本不处理来自客户端的任何请求，只负责从首领副本同步数据，保证与首领保持一致。如果首领副本发生崩溃，就会从这其中选举出一个leader。

首选首领副本：创建分区时指定的首选首领。如果不指定，则为分区的第一个副本。
follower需要从leader中同步数据，但是由于网络或者其他原因，导致数据阻塞，出现不一致的情况，为了避免这种情况，follower会向leader发送请求信息，这些请求信息中包含了follower需要数据的偏移量offset，而且这些offset是有序的。

如果有follower向leader发送了请求1，接着发送请求2，请求3，那么再发送请求4，这时就意味着follower已经同步了前三条数据，否则不会发送请求4。leader通过跟踪 每一个follower的offset来判断它们的复制进度。

默认的，如果follower与leader之间超过10s内没有发送请求，或者说没有收到请求数据，此时该follower就会被认为“不同步副本”。而持续请求的副本就是“同步副本”，当leader发生故障时，只有“同步副本”才可以被选举为leader。其中的请求超时时间可以通过参replica.lag.time.max.ms参数来配置。

我们希望每个分区的leader可以分布到不同的broker中，尽可能的达到负载均衡，所以会有一个首选首领，如果我们设置参数auto.leader.rebalance.enable为true，那么它会检查首选首领是否是真正的首领，如果不是，则会触发选举，让首选首领成为首领。



## 消费者相关的选举