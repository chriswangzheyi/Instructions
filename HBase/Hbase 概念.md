# Hbase 概念

参考资料：https://www.cnblogs.com/swordfall/p/8737328.html

##　ＨBase 数据模型

![](../Images/2.png)

- 表是行的集合。
- 行是列族的集合。
- 列族是列的集合。
- 列是键值对的集合。

## HBase 和 RDBMS的比较

![](../Images/3.png)


## 关键概念


###　Row Key 行键


与nosql数据库一样，row key是用来表示唯一一行记录的主键，HBase的数据时按照RowKey的字典顺序进行全局排序的，所有的查询都只能依赖于这一个排序维度。访问HBASE table中的行，只有三种方式：

1. 通过单个row key访问；
1. 通过row key的range（正则）
1. 全表扫描

　　Row  key 行键（Row key）可以是任意字符串(最大长度是64KB，实际应用中长度一般为10-1000bytes)，在HBASE内部，row  key保存为字节数组。存储时，数据按照Row  key的字典序(byte  order)排序存储。设计key时，要充分排序存储这个特性，将经常一起读取的行存储放到一起。(位置相关性)


