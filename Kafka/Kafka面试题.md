# Kafka面试题

## kafka的消费者是pull(拉)还是push(推)模式，这种模式有什么好处

遵循了一种大部分消息系统共同的传统的设计：producer 将消息推送到 broker，consumer 从broker 拉取消息。

优点：pull模式消费者自主决定是否批量从broker拉取数据，而push模式在无法知道消费者消费能力情况下，不易控制推送速度，太快可能造成消费者奔溃，太慢又可能造成浪费。

缺点：如果 broker 没有可供消费的消息，将导致 consumer 不断在循环中轮询，直到新消息到到达。为了避免这点，Kafka 有个参数可以让 consumer阻塞知道新消息到达(当然也可以阻塞知道消息的数量达到某个特定的量这样就可以批量发送)。


## 讲一讲 kafka 的 ack 的三种机制
request.required.acks 有三个值 0 1 -1(all)，具体如下：

* 0：生产者不会等待 broker 的 ack，这个延迟最低但是存储的保证最弱当 server 挂掉的时候就会丢数据。
* 1：服务端会等待 ack 值 leader 副本确认接收到消息后发送 ack 但是如果 leader挂掉后他不确保是否复制完成新 leader 也会导致数据丢失。
* -1(all)：服务端会等所有的 follower 的副本受到数据后才会受到 leader 发出的ack，这样数据不会丢失


## kafka 分布式（不是单机）的情况下，如何保证消息的顺序消费?
Kafka 中发送 1 条消息的时候，可以指定(topic, partition, key) 3 个参数，partiton 和 key 是可选的。

Kafka 分布式的单位是 partition，同一个 partition 用一个 write ahead log 组织，所以可以保证FIFO 的顺序。不同 partition 之间不能保证顺序。因此你可以指定 partition，将相应的消息发往同 1个 partition，并且在消费端，Kafka 保证1 个 partition 只能被1 个 consumer 消费，就可以实现这些消息的顺序消费。

另外，你也可以指定 key（比如 order id），具有同 1 个 key 的所有消息，会发往同 1 个partition，那这样也实现了消息的顺序消息

##  Kafka 消费者端的 Rebalance 操作什么时候发生？

* 同一个 consumer 消费者组 group.id 中，新增了消费者进来，会执行 Rebalance 操作
* 消费者离开当期所属的 consumer group组。比如宕机
* 分区数量发生变化时(即 topic 的分区数量发生变化时)
* 消费者主动取消订阅


## kafka 避免重复消费？

比如你处理的数据要写库（mysql，redis等），你先根据主键查一下，如果这数据都有了，你就别插入了，进行一些消息登记或者update等其他操作。另外，数据库层面也可以设置唯一健，确保数据不要重复插入等 。一般这里要求生产者在发送消息的时候，携带全局的唯一id。

## kafka维护消息状态的跟踪方法

Kafka中的Topic 被分成了若干分区，每个分区在同一时间只被一个 consumer 消费。然后再通过offset进行消息位置标记，通过位置偏移来跟踪消费状态。相比其他一些消息队列使用“一个消息被分发到consumer 后 broker 就马上进行标记或者等待 customer 的通知后进行标记”的优点是，避免了通信消息发送后，可能出现的程序奔溃而出现消息丢失或者重复消费的情况。同时也无需维护消息的状态，不用加锁，提高了吞吐量。


## kafka是怎么样的？

Kafka 将消息以 topic 为单位进行归纳

将向 Kafka topic 发布消息的程序成为 producers.

将预订 topics 并消费消息的程序成为 consumer.

Kafka 以集群的方式运行，可以由一个或多个服务组成，每个服务叫做一个 broker.

producers 通过网络将消息发送到 Kafka 集群，集群向消费者提供消息

## 数据传输的事物定义有哪三种？

数据传输的事务定义通常有以下三种级别：

（1）最多一次: 消息不会被重复发送，最多被传输一次，但也有可能一次不传输

（2）最少一次: 消息不会被漏发送，最少被传输一次，但也有可能被重复传输.

（3）精确的一次（Exactly once）: 不会漏传输也不会重复传输,每个消息都传输被一次而且仅仅被传输一次，这是大家所期望的

## Kafka 判断一个节点是否还活着有那两个条件？

（1）节点必须可以维护和 ZooKeeper 的连接，Zookeeper 通过心跳机制检查每个节点的连
接

（2）如果节点是个 follower,他必须能及时的同步 leader 的写操作，延时不能太久

## producer 是否直接将数据发送到 broker 的 leader(主节点)？

producer 直接将数据发送到 broker 的 leader(主节点)，不需要在多个节点进行分发，为了帮助 producer 做到这点，所有的 Kafka 节点都可以及时的告知:哪些节点是活动的，目标topic 目标分区的 leader 在哪。这样 producer 就可以直接将消息发送到目的地了。

## Kafa consumer 是否可以消费指定分区消息？

Kafa consumer 消费消息时，向 broker 发出"fetch"请求去消费特定分区的消息，consumer指定消息在日志中的偏移量（offset），就可以消费从这个位置开始的消息，customer 拥有了 offset 的控制权，可以向后回滚去重新消费之前的消息，这是很有意义的

## Kafka中的ISR(InSyncRepli)、OSR(OutSyncRepli)、AR(AllRepli)代表什么？ISR的伸缩又指什么

* ISR : 速率和leader相差低于10秒的follower的集合
* OSR : 速率和leader相差大于10秒的follower
* AR : 全部分区的follower

ISR是由leader维护，follower从leader同步数据有一些延迟（包括延迟时间replica.lag.time.max.ms和延迟条数replica.lag.max.messages两个维度, 当前最新的版本0.10.x中只支持

replica.lag.time.max.ms这个维度），任意一个超过阈值都会把follower剔除出ISR, 存入OSR（Outof-Sync Replicas）列表，新加入的follower也会先存放在OSR中。AR=ISR+OSR。


## Kafka 数据一致性原理




## 解释偏移的作用。

给分区中的消息提供了一个顺序ID号，我们称之为偏移量。因此，为了唯一地识别分区中的每条消息，我们使用这些偏移量。

## kafka中的 zookeeper 起到什么作用，可以不用zookeeper么

zookeeper 是一个分布式的协调组件，早期版本的kafka用zk做meta信息存储，consumer的消费状态，group的管理以及 offset的值。考虑到zk本身的一些因素以及整个架构较大概率存在单点问题，新版本中逐渐弱化了zookeeper的作用。新的consumer使用了kafka内部的group coordination协议，也减少了对zookeeper的依赖，

但是broker依然依赖于ZK，zookeeper 在kafka中还用来选举controller 和 检测broker是否存活等等。

## kafka follower如何与leader同步数据

Kafka的复制机制既不是完全的同步复制，也不是单纯的异步复制。完全同步复制要求All Alive Follower都复制完，这条消息才会被认为commit，这种复制方式极大的影响了吞吐率。而异步复制方式下，Follower异步的从Leader复制数据，数据只要被Leader写入log就被认为已经commit，这种情况下，如果leader挂掉，会丢失数据，kafka使用ISR的方式很好的均衡了确保数据不丢失以及吞吐率。Follower可以批量的从Leader复制数据，而且Leader充分利用磁盘顺序读以及send file(zero copy)机制，这样极大的提高复制性能，内部批量写磁盘，大幅减少了Follower与Leader的消息量差。

## kafka 为什么那么快


Cache Filesystem Cache PageCache缓存	
	
顺序写 由于现代的操作系统提供了预读和写技术，磁盘的顺序写大多数情况下比随机写内存还要快。
	
Zero-copy 零拷技术减少拷贝次数
	
Batching of Messages 批量量处理。合并小的请求，然后以流的方式进行交互，直顶网络上限。
	
Pull 拉模式 使用拉模式进行消息的获取消费，与消费端处理能力相符。
