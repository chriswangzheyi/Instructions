# Kylin 面试题

## Kylin的优点和缺点

优点:预计算，界面可视化

缺点：依赖较多，属于重量级方案，运维成本很高

不适合做即席查询

预计算量大，非常消耗资源

## Kylin的rowkey如何设计

将过滤频率较高的列放置在过滤频率较低的列之前

将基数高的列放置在基数低的列之前

在查询中被用作过滤条件的维度有可能放在其他维度的前面


## cuboid,cube和segment的关系

Cube是所有cubiod的组合，一个cube包含一个或者多个cuboid

Cuboid 在 Kylin 中特指在某一种维度组合下所计算的数据。

Cube Segment 是指针对源数据中的某一片段，全量构建的cube只存在唯一的segment，该segment没有分割时间的概念，增量构建的cube，不同时间的数据分布在不同的segment中

## kylin你一般怎么调优

### Cube调优

l剪枝优化(衍生维度，聚合组，强制维度，层级维度，联合维度)

l并发粒度优化

lRowkeys优化(编码，按维度分片，调整维度顺序)

l降低度量精度

l及时清理无用的segment

 

### Rowkey调优

lKylin rowkey的编码和压缩选择

l维度在rowkey中顺序的调整，

l将过滤频率较高的列放置在过滤频率较低的列之前，

l将基数高的列放置在基数低的列之前。

l在查询中被用作过滤条件的维度有可能放在其他维度的前面。

充分利用过滤条件来缩小在HBase中扫描的范围， 从而提高查询的效率。 


## 为什么kylin的维度不建议过多？

Cube 的最大物理维度数量 (不包括衍生维度) 是 63，但是不推荐使用大于 30 个维度的 Cube，会引起维度灾难。

## Kylin cube的构建过程是怎么样的

1. 选择model
1. 选择维度
1. 选择指标
1. cube设计(包括维度和rowkeys)
1. 构建cube(mr程序，hbase存储元数据信息及计算好的数据信息)



