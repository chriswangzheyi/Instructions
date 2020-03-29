# Kafka offset管理

## 背景

Kafka中的每个partition都由一系列有序的、不可变的消息组成，这些消息被连续的追加到partition中。partition中的每个消息都有一个连续的序号，用于partition唯一标识一条消息。

**Offset记录着下一条将要发送给Consumer的消息的序号。**

Offset从语义上来看拥有两种：Current Offset和Committed Offset。

## Current Offset

Current Offset保存在Consumer客户端中，它表示Consumer希望收到的下一条消息的序号。它仅仅在poll()方法中使用。例如，Consumer第一次调用poll()方法后收到了20条消息，那么Current Offset就被设置为20。这样Consumer下一次调用poll()方法时，Kafka就知道应该从序号为21的消息开始读取。这样就能够保证每次Consumer poll消息时，都能够收到不重复的消息。

## Committed Offset

Committed Offset保存在Broker上，它表示Consumer已经确认消费过的消息的序号。主要通过commitSync和commitAsyncAPI来操作。

举个例子，Consumer通过poll() 方法收到20条消息后，此时Current Offset就是20，经过一系列的逻辑处理后，并没有调用consumer.commitAsync()或consumer.commitSync()来提交Committed Offset，那么此时Committed Offset依旧是0。

Committed Offset主要用于Consumer Rebalance。在Consumer Rebalance的过程中，一个partition被分配给了一个Consumer，那么这个Consumer该从什么位置开始消费消息呢？答案就是Committed Offset。另外，如果一个Consumer消费了5条消息（poll并且成功commitSync）之后宕机了，重新启动之后它仍然能够从第6条消息开始消费，因为Committed Offset已经被Kafka记录为5。

**总结一下，Current Offset是针对Consumer的poll过程的，它可以保证每次poll都返回不重复的消息；而Committed Offset是用于Consumer Rebalance过程的，它能够保证新的Consumer能够从正确的位置开始消费一个partition，从而避免重复消费。**


在Kafka 0.9前，Committed Offset信息保存在zookeeper的[consumers/{group}/offsets/{topic}/{partition}]目录中（zookeeper其实并不适合进行大批量的读写操作，尤其是写操作）。而在0.9之后，所有的offset信息都保存在了Broker上的一个名为__consumer_offsets的topic中。

Kafka集群中offset的管理都是由Group Coordinator中的Offset Manager完成的。


## Group Coordinator

Group Coordinator是运行在Kafka集群中每一个Broker内的一个进程。它主要负责Consumer Group的管理，Offset位移管理以及Consumer Rebalance。

对于每一个Consumer Group，Group Coordinator都会存储以下信息：

- 订阅的topics列表
- Consumer Group配置信息，包括session timeout等
- 组中每个Consumer的元数据。包括主机名，consumer id
- 每个Group正在消费的topic partition的当前offsets
- Partition的ownership元数据，包括consumer消费的partitions映射关系






