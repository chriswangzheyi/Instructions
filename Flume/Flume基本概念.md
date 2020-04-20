# Flume基本概念

---

## 三大组件

### Source （收集信息）

完成对日志数据的收集，分成transtion 和 even 打入到 channel 中

### Channel  （传递信息）

主要提供一个队列的功能，对 source 提供中的数据进行简单的缓存

### Sink （存储信息）

取出 channel 中的数据，进行相应的存储文件系统，数据库，或者提交到远程服务器


## 逻辑结构

![](../Images/1.png)

Flume 逻辑上分三层架构：

Agent，collector，storage

### Agent

用于采集数据，agent 是 flume 中存储数据流的地方，同时 agent 会将产生的数据传输到 collector

### Collector

Collector 的作用是坚多个 agent 的数据汇总后，加载到 storage 中，多个 collector 之间遵循负载均衡规则

### Storage

Storage 是存储系统，可以是一个普通 file，也可以是 HDFS，HIVE

### Master

Master 是管理协调 agent 和 collector 的配置等信息，是 flume 集群的控制器。

在 flume 中，最重要的抽象是 data flow （数据流），data flow 描述了数据从生产，传输、处理并追踪写入目标的一条路径。


### 多collector性能

多个 collector 能够增加日志收集的吞吐量，提高 collector 的有效性能够提高数据的传输速度，数据的收集是可并行的，此外，来自多个 agent 的数据能够分配到多个 collector 上加载。

### 多collector下agent的划分

前面的图展示flume节点典型的拓扑结构和数据流，为了可靠的传输，当collector停止运行或是失去与agents的联系的时候，agents将会 存储他们的events在各自的本地硬盘上，这些agents试图重新连接collector，因为collector的宕机，任何处理和分析的数据流都 被阻塞。

![](../Images/2.png)

当你有多个collector如上图所示，即使在collector宕机的情况下，数据处理仍然能够进行下去，如果 collector b 宕机了，agent a，agent b，ageng e，和agentf会分别继续传送events通过collector a 和collector c，agent c 和agent d 的不得不排在其他agent的后面等待日志的处理直到collector b重新上线。
