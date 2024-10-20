# Kafka partition的数量问题

kafka的每个topic都可以创建多个partition，partition的数量无上限，并不会像replica一样受限于broker的数量，因此partition的数量可以随心所欲的设置。那确定partition的数量就需要思考一些权衡因素。


- 每个Partition只会在一个Broker上，物理上每个Partition对应的是一个文件夹
- Kafka默认使用的是hash进行分区，所以会出现不同的分区数据不一样的情况，但是partitioner是可以override的
- Partition包含多个Segment，每个Segment对应一个文件，Segment可以手动指定大小，当Segment达到阈值时，将不再写数据，每个Segment都是大小相同的
- Segment由多个不可变的记录组成，记录只会被append到Segment中，不会被单独删除或者修改，每个Segment中的Message数量不一定相等

![](../Images/10.png)

## 越多的partition可以提供更高的吞吐量

在kafka中，单个partition是kafka并行操作的最小单元。每个partition可以独立接收推送的消息以及被consumer消费，相当于topic的一个子通道，partition和topic的关系就像高速公路的车道和高速公路的关系一样，起始点和终点相同，每个车道都可以独立实现运输，不同的是kafka中不存在车辆变道的说法，入口时选择的车道需要从一而终。而kafka的吞吐量显而易见，在资源足够的情况下，partition越多速度越快。

这里提到的资源充足解释一下，假设我现在一个partition的最大传输速度为p，目前kafka集群共有三个broker，每个broker的资源足够支撑三个partition最大速度传输，那我的集群最大传输速度为3*3*p=9p，假设在不增加资源的情况下将partition增加到18个，每个partition只能以p/2的速度传输数据，因此传输速度上限还是9p，并不能再提升，因此吞吐量的设计需要考虑broker的资源上限。当然，kafka跟其他集群一样，可以横向扩展，再增加三个相同资源的broker，那传输速度即可达到18p。

## 越多的分区需要打开更多的文件句柄

在kafka的broker中，每个分区都会对照着文件系统的一个目录。

在kafka的数据日志文件目录中，每个日志数据段都会分配两个文件，一个索引文件和一个数据文件。因此，随着partition的增多，需要的文件句柄数急剧增加，必要时需要调整操作系统允许打开的文件句柄数。

## 更多的分区会导致端对端的延迟

kafka端对端的延迟为producer端发布消息到consumer端消费消息所需的时间，即consumer接收消息的时间减去produce发布消息的时间。kafka在消息正确接收后才会暴露给消费者，即在保证in-sync副本复制成功之后才会暴露，瓶颈则来自于此。在一个broker上的副本从其他broker的leader上复制数据的时候只会开启一个线程，假设partition数量为n，每个副本同步的时间为1ms，那in-sync操作完成所需的时间即n*1ms，若n为10000，则需要10秒才能返回同步状态，数据才能暴露给消费者，这就导致了较大的端对端的延迟。

## 越多的partition意味着需要更多的内存

在新版本的kafka中可以支持批量提交和批量消费，而设置了批量提交和批量消费后，每个partition都会需要一定的内存空间。假设为100k，当partition为100时，producer端和consumer端都需要10M的内存；当partition为100000时，producer端和consumer端则都需要10G内存。无限的partition数量很快就会占据大量的内存，造成性能瓶颈。


## 越多的partition会导致更长时间的恢复期

kafka通过多副本复制技术，实现kafka的高可用性和稳定性。每个partition都会有多个副本存在于多个broker中，其中一个副本为leader，其余的为follower。当kafka集群其中一个broker出现故障时，在这个broker上的leader会需要在其他broker上重新选择一个副本启动为leader，这个过程由kafka controller来完成，主要是从Zookeeper读取和修改受影响partition的一些元数据信息。

通常情况下，当一个broker有计划的停机上，该broker上的partition leader会在broker停机前有次序的一一移走，假设移走一个需要1ms，10个partition leader则需要10ms，这影响很小，并且在移动其中一个leader的时候，其他九个leader是可用的，因此实际上每个partition leader的不可用时间为1ms。但是在宕机情况下，所有的10个partition

leader同时无法使用，需要依次移走，最长的leader则需要10ms的不可用时间窗口，平均不可用时间窗口为5.5ms，假设有10000个leader在此宕机的broker上，平均的不可用时间窗口则为5.5s。

更极端的情况是，当时的broker是kafka controller所在的节点，那需要等待新的kafka leader节点在投票中产生并启用，之后新启动的kafka leader还需要从zookeeper中读取每一个partition的元数据信息用于初始化数据。在这之前partition leader的迁移一直处于等待状态。

## 总结

通常情况下，越多的partition会带来越高的吞吐量，但是同时也会给broker节点带来相应的性能损耗和潜在风险，虽然这些影响很小，但不可忽略，因此需要根据自身broker节点的实际情况来设置partition的数量以及replica的数量。
