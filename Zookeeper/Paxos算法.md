# Paxos算法

---

## 定义

Paxos算法是用来解决分布式系统中，如何就某个值达成一致的算法。


### 分类

![](../Images/3.png)


### 角色

Proposer：议案发起者。

Acceptor：决策者，可以批准议案。

Learner：最终决策的学习者。

![](../Images/1.png)


## Paxos算法的过程 (Basic Paxos)


### Prepare阶段

Proposer选择一个提案编号n，发送Prepare(n)请求给超过半数（或更多）的Acceptor。

Acceptor收到消息后，如果n比它之前见过的编号大，就回复这个消息，而且以后不会接受小于n的提案。另外，如果之前已经接受了小于n的提案，回复那个提案编号和内容给Proposer。


### Accept阶段

当Proposer收到超过半数的回复时，就可以发送Accept(n, value)请求了。 n就是自己的提案编号，value是Acceptor回复的最大提案编号对应的value，如果Acceptor没有回复任何提案，value就是Proposer自己的提案内容。

Acceptor收到消息后，如果n大于等于之前见过的最大编号，就记录这个提案编号和内容，回复请求表示接受。
当Proposer收到超过半数的回复时，说明自己的提案已经被接受。否则回到第一步重新发起提案。

![](../Images/2.png)


###三种情况

####情况1：提案已接受

X、Y代表客户端，S1到S5是服务端，既代表Proposer又代表Acceptor。为了防止重复，Proposer提出的编号由两部分组成：

序列号.Server ID

例如S1提出的提案编号，就是1.1、2.1、3.1……

![](../Images/4.png)

这个过程表示，S1收到客户端的提案X，于是S1作为Proposer，给S1-S3发送Prepare(3.1)请求，由于Acceptor S1-S3没有接受过任何提案，所以接受该提案。然后Proposer S1-S3发送Accept(3.1, X)请求，提案X成功被接受。

在提案X被接受后，S5收到客户端的提案Y，S5给S3-S5发送Prepare(4.5)请求。对S3来说，4.5比3.1大，且已经接受了X，它会回复这个提案 (3.1, X)。S5收到S3-S5的回复后，使用X替换自己的Y，于是发送Accept(4.5, X)请求。S3-S5接受提案。最终所有Acceptor达成一致，都拥有相同的值X。

这种情况的结果是：新Proposer会使用已接受的提案。


#### 情况2：提案未接受，新Proposer可见

![](../Images/5.png)

S3接受了提案(3.1, X)，但S1-S2还没有收到请求。此时S3-S5收到Prepare(4.5)，S3会回复已经接受的提案(3.1, X)，S5将提案值Y替换成X，发送Accept(4.5, X)给S3-S5，对S3来说，编号4.5大于3.1，所以会接受这个提案。

然后S1-S2接受Accept(3.1, X)，最终所有Acceptor达成一致。

这种情况的结果是：新Proposer会使用已提交的值，两个提案都能成功



#### 情况3：提案未接受，新Proposer不可见

![](../Images/6.png)

1接受了提案(3.1, X)，S3先收到Prepare(4.5)，后收到Accept(3.1, X)，由于3.1小于4.5，会直接拒绝这个提案。所以提案X无法收到超过半数的回复，这个提案就被阻止了。提案Y可以顺利通过。

这种情况的结果是：新Proposer使用自己的提案，旧提案被阻止



## Paxos算法的过程 (Multi-Paxos)


### 1. Leader选举

一个最简单的选举方法，就是Server ID最大的当Leader。

每个Server间隔T时间向其他Server发送心跳包，如果一个Server在2T时间内没有收到来自更高ID的心跳，那么它就成为Leader。

其他Proposer，必须拒绝客户端的请求，或将请求转发给Leader。


### 2. 省略Prepare阶段

Prepare的作用是阻止旧的提案，以及检查是否有已接受的提案值。

当只有一个Leader发送提案的时候，Prepare是不会产生冲突的，可以省略Prepare阶段，这样就可以减少一半RPC请求。

Prepare请求的逻辑修改为：

Acceptor记录一个全局的最大提案编号

回复最大提案编号，如果当前entry以及之后的所有entry都没有接受任何提案，回复noMoreAccepted

当Leader收到超过半数的noMoreAccepted回复，之后就不需要Prepare阶段了，只需要发送Accept请求。直到Accept被拒绝，就重新需要Prepare阶段

