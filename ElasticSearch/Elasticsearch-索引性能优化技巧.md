# Elasticsearch-索引性能优化技巧

参考资料：https://blog.csdn.net/gongpulin/article/details/78766097

## 优化索引

### 对响应速度敏感

把每个索引的 index.refresh_interval 改到 30s

### 对响应速度不敏感

设置 index.number_of_replicas: 0关闭副本


## 优化储存

磁盘在现代服务器上通常都是瓶颈。Elasticsearch 重度使用磁盘，你的磁盘能处理的吞吐量越大，你的节点就越稳定。

1.使用 SSD

2.使用 RAID 0


## 优化内存占用

1. 删除不用的索引

2.关闭索引（文件存在与磁盘，只是释放掉内存）



