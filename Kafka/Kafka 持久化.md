# Kafka 持久化

## 概念

一个 Topic 被分成多 Partition，每个 Partition 在存储层面是一个 append-only 日志文件，属于一个 Partition 的消息都会被直接追加到日志文件的尾部，每条消息在文件中的位置称为 offset（偏移量）。

![](../Images/11.png)

日志文件由“日志条目（log entries）”序列组成，每一个日志条目包含一个4字节整型数（值为N），其后跟N个字节的消息体。每条消息都有一个当前 Partition 下唯一的64字节的 offset，标识这条消息的起始位置。消息格式如下

	On-disk format of a message
	
	offset         : 8 bytes 
	message length : 4 bytes (value: 4 + 1 + 1 + 8(if magic value > 0) + 4 + K + 4 + V)
	crc            : 4 bytes
	magic value    : 1 byte
	attributes     : 1 byte
	timestamp      : 8 bytes (Only exists when magic value is greater than zero)
	key length     : 4 bytes
	key            : K bytes
	value length   : 4 bytes
	value          : V bytes


##写
日志文件允许串行附加，并且总是附加到最后一个文件。当文件达到配置指定的大小（log.segment.bytes = 1073741824 (bytes)）时，就会被滚动到一个新文件中（每个文件称为一个 segment file）。日志有两个配置参数：M，强制操作系统将文件刷新到磁盘之前写入的消息数；S，强制操作系统将文件刷新到磁盘之前的时间（秒）。在系统崩溃的情况下，最多会丢失M条消息或S秒的数据。

##读
通过给出消息的偏移量（offset）和最大块大小（S）来读取数据。返回一个缓冲区为S大小的消息迭代器，S应该大于任何单个消息的大小，如果消息异常大，则可以多次重试读取，每次都将缓冲区大小加倍，直到成功读取消息为止。可以指定最大消息大小和缓冲区大小，以使服务器拒绝大于某个大小的消息。读取缓冲区可能以部分消息结束，这很容易被大小分隔检测到。

读取指定偏移量的数据时，需要首先找到存储数据的 segment file，由全局偏移量计算 segment file 中的偏移量，然后从此位置开始读取。

##删除
消息数据随着 segment file 一起被删除。Log manager 允许可插拔的删除策略来选择哪些文件符合删除条件。当前策略为删除修改时间超过 N 天前的任何日志，或者是保留最近的 N GB 的数据。

为了避免在删除时阻塞读操作，采用了 copy-on-write 技术：删除操作进行时，读取操作的二分查找功能实际是在一个静态的快照副本上进行的。
