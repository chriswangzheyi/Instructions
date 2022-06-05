# Hive调优

##化的核心思想

* 减少数据量（例如分区、列剪裁）
* 避免数据倾斜（例如加参数、Key打散）
* 避免全表扫描（例如on添加加上分区等）
* 减少job数（例如相同的on条件的join放在一起作为一个任务）

## 大小表的join

写有Join操作的查询语句时有一条原则：应该将条目少的表/子查询放在Join操作符的左边。原因是在Join操作的Reduce阶段，位于Join操作符左边的表的内容会被加载进内存，将条目少的表放在左边，可以有效减少发生OOM错误的几率。

但新版的hive已经对小表JOIN大表和大表JOIN小表进行了优化。小表放在左边和右边已经没有明显区别。

不过在做join的过程中通过小表在前可以适当地减少数据量，提高效率

## 优化SQL处理join数据倾斜

种情况很常见，比如当事实表是日志类数据时，往往会有一些项没有记录到，我们视情况会将它置为null，或者空字符串、-1等。如果缺失的项很多，在做join时这些空值就会非常集中，拖累进度。因此，若不需要空值数据，就提前写where语句过滤掉。需要保留的话，将空值key用随机方式打散，例如将用户ID为null的记录随机改为负值.

## 调整mapper数量

mapper数量与输入文件的split数息息相关，在Hadoop源码org.apache.hadoop.mapreduce.lib.input.FileInputFormat类中可以看到split划分的具体逻辑。

## 调整reducer数量

reducer数量的确定方法比mapper简单得多。使用参数mapred.reduce.tasks可以直接设定reducer数量，不像mapper一样是期望值。但如果不设这个参数的话，Hive就会自行推测，

## 合并小文件

## 列裁剪和分区裁剪

列裁剪就是在查询时只读取需要的列，分区裁剪就是只读取需要的分区。

## 谓词下推

### sort by代替order by

HiveQL中的order by与其他SQL方言中的功能一样，就是将结果按某字段全局排序，这会导致所有map端数据都进入一个reducer中，在数据量大时可能会长时间计算不完。如果使用sort by，那么还是会视情况启动多个reducer进行排序，并且保证每个reducer内局部有序。


### group by代替distinct

当要统计某一列的去重数时，如果数据量很大，count(distinct)就会非常慢，原因与order by类似，count(distinct)逻辑只会有很少的reducer来处理。这时可以用group by来改写

### map端预聚合

group by时，如果先起一个combiner在map端做部分预聚合，可以有效减少shuffle数据量。预聚合的配置项是hive.map.aggr，默认值true，对应的优化器为GroupByOptimizer，简单方便。通过hive.groupby.mapaggr.checkinterval参数也可以设置map端预聚合的行数阈值，超过该值就会分拆job，默认值100000。


###分桶表map join

map join对分桶表还有特别的优化。由于分桶表是基于一列进行hash存储的，因此非常适合抽样（按桶或按块抽样）。