# Flink State

## 定义

一般指一个具体的task/operator某时刻在内存中的状态(例如某属性的值)

### 与Checkpoint的区别

checkpoint表示了一个Flink Job，在一个特定时刻的一份全局状态快照，即包含了一个job下所有task/operator某时刻的状态。