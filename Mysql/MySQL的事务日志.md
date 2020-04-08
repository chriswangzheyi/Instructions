# MySQL的事务日志

参考：https://www.cnblogs.com/f-ck-need-u/archive/2018/05/08/9010872.html

---

innodb事务日志包括redo log和undo log。redo log是重做日志，提供前滚操作，undo log是回滚日志，提供回滚操作。

undo log不是redo log的逆向过程，其实它们都算是用来恢复的日志：

1.redo log通常是物理日志，记录的是数据页的物理修改，而不是某一行或某几行修改成怎样怎样，它用来恢复提交后的物理数据页(恢复数据页，且只能恢复到最后一次提交的位置)。

2.undo用来回滚行记录到某个版本。undo log一般是逻辑日志，根据每行记录进行记录。

## redo log


 在innoDB的存储引擎中，事务日志通过重做(redo)日志和innoDB存储引擎的日志缓冲(InnoDB Log Buffer)实现。事务开启时，事务中的操作，都会先写入存储引擎的日志缓冲中，在事务提交之前，这些缓冲的日志都需要提前刷新到磁盘上持久化，这就是DBA们口中常说的“日志先行”(Write-Ahead Logging)。当事务提交之后，在Buffer Pool中映射的数据文件才会慢慢刷新到磁盘。此时如果数据库崩溃或者宕机，那么当系统重启进行恢复时，就可以根据redo log中记录的日志，把数据库恢复到崩溃前的一个状态。未完成的事务，可以继续提交，也可以选择回滚，这基于恢复的策略而定。

在系统启动的时候，就已经为redo log分配了一块连续的存储空间,以顺序追加的方式记录Redo Log,通过顺序IO来改善性能。所有的事务共享redo log的存储空间，它们的Redo Log按语句的执行顺序，依次交替的记录在一起。如下一个简单示例：

        记录1：<trx1, insert...>

        记录2：<trx2, delete...>

        记录3：<trx3, update...>

        记录4：<trx1, update...>

        记录5：<trx3, insert...>


## undo log

 undo log主要为事务的回滚服务。在事务执行的过程中，除了记录redo log，还会记录一定量的undo log。undo log记录了数据在每个操作前的状态，如果事务执行过程中需要回滚，就可以根据undo log进行回滚操作。单个事务的回滚，只会回滚当前事务做的操作，并不会影响到其他的事务做的操作。

以下是undo+redo事务的简化过程

假设有2个数值，分别为A和B,值为1，2

        1. start transaction;

        2. 记录 A=1 到undo log;

        3. update A = 3；

        4. 记录 A=3 到redo log；

        5. 记录 B=2 到undo log；

        6. update B = 4；

        7. 记录B = 4 到redo log；

        8. 将redo log刷新到磁盘

        9. commit

        在1-8的任意一步系统宕机，事务未提交，该事务就不会对磁盘上的数据做任何影响。如果在8-9之间宕机，恢复之后可以选择回滚，也可以选择继续完成事务提交，因为此时redo log已经持久化。若在9之后系统宕机，内存映射中变更的数据还来不及刷回磁盘，那么系统恢复之后，可以根据redo log把数据刷回磁盘。

        所以，redo log其实保障的是事务的持久性和一致性，而undo log则保障了事务的原子性。




