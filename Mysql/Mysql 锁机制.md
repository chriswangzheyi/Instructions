# Mysql 锁机制

## MySQL 锁概述

### Mysql锁的类型

表级锁：开销小，加锁快；不会出现死锁；锁定粒度大，发生锁冲突的概率最高，并发度最低。 

行级锁：开销大，加锁慢；会出现死锁；锁定粒度最小，发生锁冲突的概率最低，并发度也最高。 

页面锁：开销和加锁时间界于表锁和行锁之间；会出现死锁；锁定粒度界于表锁和行锁之间，并发度一般 


## 表级锁

表锁的语法是lock tables … read/write。可以用unlock tables主动释放锁，也可以在客户端断开的时候自动释放。

## 行锁

行锁就是针对数据表中行记录的锁。比如事务A更新了一行，而这时候事务B也要更新同一行，则必须等事务A的操作完成后才能进行更新
