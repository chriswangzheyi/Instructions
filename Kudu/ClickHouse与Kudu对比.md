# ClickHouse与Kudu对比

参考：https://zhuanlan.zhihu.com/p/597055907

|  方面  | Kudu | ClickHouse |
|  ----  | ----  |----  |
| 架构设计  | 单元格 |单元格 |
| 元数据管理 | Kudu是Master-slave的架构，Master挂掉不能工作 |Clickhouse每台Server的地位都是等价的，是multi-master模式。解决单点故障。 |
| 元数据管理 | Kudu使用Master Server管理元数据。 |ClickHouse使用Zookeeper管理元数据。 |
| SQL支持 | Kudu不支持标准SQL,有put，get等api代码操作；与Impala整合后支持SQL操作。 |ClickHouse对于标准SQL的支持相对完好。 |
| 应用场景 | Kudu应用主要是随机读写且兼容大批量读取操作场景，生产中经常与Impala集成，也可做OLAP分析。 |Clickhouse应用场景主要是实时OLAP分析。不是太擅长随机读数据。|
| 数据CRUD | Kudu支持数据更新，删除操作，可以通过api代码实现，也可以通过与impala整合SQL实现；仅支持单条数据的事务。Kudu对数据快速读取和快速插入数据的场景支持比较好，原子数据查询延迟低，与Impala整合可以做OLAP操作。 |ClickHouse是分析型列式数据库，处理的数据一般不变化，变化一般也不会更新，对于update,delete的支持比较脆弱，实际上clickhouse不支持标准的update和delete操作，通过alter操作实现；不支持事务。ClickHouse最好大批量插入数据，对数据原子行大量读取，效率不高，延迟大，主要做OLAP分析操作 |
| 扩展性 | Kudu由于Tablet Server的特殊结构，扩展性差，支持300个节点。 |ClickHouse集群节点无上限 |

