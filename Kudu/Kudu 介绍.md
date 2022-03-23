# Kudu 介绍

## 简介

Kudu 是为 Apache Hadoop 平台开发的列式存储管理器。Kudu 具有 Hadoop 生态系统应用程序的共同技术属性：它在商品硬件上运行，具有水平可扩展性，并支持高可用操作。简单来说：kudu是一个与Hbase类似的列式存储分布式数据库。

## 为什么需要kudu？

HDFS 与HBase的数据存储的缺点目前数据存储有了HDFS与HBase，为什么还要额弄一个kudu呢？

### HDFS

使用列式存储格式Apache Parquet , Apache ORC，适合离线分析，不支持单条记录级别的update操作，随机读写能力差

### HBase

可以进行高效读写，却并不是适合基于SQL的数据分析方向，大批量数据获取的性能差。

### Kudu

kudu较好的解决了HDFS与HBase的这些特点，它不及HDFS批处理快，也不及HBase随机读写能力强，但反过来它比HBase批处理快,而且比HDFS随机读写能力强（适合实时写入或这更新场景频繁的场景）.这就是他能解决的问题。