# 对于 Spark 中的数据倾斜问题你有什么好的方案

1前提是定位数据倾斜，是 OOM 了，还是任务执行缓慢，看日志，看 WebUI

2解决方法，有多个方面:

* 避免不必要的 shuffle，如使用广播小表的方式，将 reduce-side-join 提升为 map-side-join
* 分拆发生数据倾斜的记录，分成几个部分进行，然后合并 join 后的结果
* 改变并行度，可能并行度太少了，导致个别 task 数据压力大
* 两阶段聚合，先局部聚合，再全局聚合
* 自定义 paritioner，分散 key 的分布，使其更加均匀