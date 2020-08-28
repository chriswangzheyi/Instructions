# Netty 三大特性之Selector

## 介绍

使用一个线程，处理多个客户端连接。 Selector能够坚持多个注册通道上是否有事件发生（注意：多个channel以事件的方式可以注册到同一个selector），如果有事件发生，便获取事件然后针对每个事件进行相应的处理。这样就可以只用一个单线程去管理多个通道。

只有在连接有事件发生时，才会进行读写，大大减少了系统的开销。

避免了多线程之间的上下文切换导致的开销。


## 流程

1. 当客户端连接时，会通过ServerSocketChannel 得到 SocketChannel
1. Selector 进行监听  select 方法, 返回有事件发生的通道的个数.

1. 将socketChannel注册到Selector上, register(Selector sel, int ops), 一个selector上可以注册多个SocketChannel

1. 注册后返回一个 SelectionKey, 会和该Selector 关联(集合)
1. 进一步得到各个 SelectionKey (有事件发生)
1. 在通过 SelectionKey  反向获取 SocketChannel , 方法 channel()
1. 可以通过  得到的 channel  , 完成业务处理

