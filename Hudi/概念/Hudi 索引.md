# Hudi 索引

## 索引的概念

Hudi 分区由多个 File Group 构成，每个 File Group 由 File ID 进行标识。File Group 内的文件分为 Base File ( parquet 格式) 和 Delta File( log 文件)，Delta File 记录对 Base File 的修改。Hudi 使用了 MVCC 的设计，可以通过 Compaction 任务把 Delta File 和 Base File 合并成新的 Base File，并通过 Clean 操作删除不需要的旧文件。

Hudi 通过索引机制将给定的 Hudi 记录一致地映射到 File ID，从而提供高效的 Upsert。Record Key 和 File Group/File ID 之间的这种映射关系，一旦在 Record 的第一个版本确定后，就永远不会改变。简而言之，包含一组记录的所有版本必然在同一个 File Group 中。

## 索引的类型

* Bloom 索引（默认）：对 record key 创建布隆过滤器
* Simple 索引：对update/delete 操作和存储中提取出来的key，执行轻量级的 join
* HBase 索引 ：通过外部的 HBase存储来管理索引

## Global index 和 Non Global index

Global index(全局索引): Global index 要求保证 key 在表中所有分区的都是唯一的，保证一个给定的 record key 在表中只能找能唯一的一条数据。Global index 提供了强唯一性保证，但是随着表增大，update/delete 操作损失的性能越高，因此只适用于小表。

Non Global index(非全局索引)：非全局索引只能保证数据在分区的唯一性。但是通过对 Hudi 索引的学习，不难了解到 key 与 file id 存在映射关系，同一个 key 的数据（包括 updete/delete）必然会存在同一个分区里面。这种实现方式提供更好的索引查找性能，适用于大表。



