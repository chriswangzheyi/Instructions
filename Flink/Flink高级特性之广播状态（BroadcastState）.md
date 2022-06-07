# Flink高级特性之广播状态（BroadcastState）

参考：https://blog.csdn.net/yang_shibiao/article/details/122739655

## BroadcastState介绍

如果遇到需要下发/广播配置、规则等低吞吐事件流到下游所有 task 时，就可以使用 Broadcast State。Broadcast State 是 Flink 1.5 引入的新特性。

下游的 task 接收这些配置、规则并保存为 BroadcastState, 将这些配置应用到另一个数据流的计算中 。

### 场景举例：

动态更新计算规则: 如事件流需要根据最新的规则进行计算，则可将规则作为广播状态广播到下游Task中。

实时增加额外字段: 如事件流需要实时增加用户的基础信息，则可将用户的基础信息作为广播状态广播到下游Task中。

### API介绍：

首先创建一个Keyed 或Non-Keyed 的DataStream，然后再创建一个BroadcastedStream，最后通过DataStream来连接(调用connect 方法)到Broadcasted Stream 上，这样实现将BroadcastState广播到Data Stream 下游的每个Task中。


## 案例

### 案例一：

 如果DataStream是Keyed Stream ，则连接到Broadcasted Stream 后， 添加处理ProcessFunction 时需要使用KeyedBroadcastProcessFunction 来实现， 下面是KeyedBroadcastProcessFunction 的API，代码如下所示：

	public abstract class KeyedBroadcastProcessFunction<KS, IN1, IN2, OUT> extends BaseBroadcastProcessFunction {
	    public abstract void processElement(final IN1 value, final ReadOnlyContext ctx, final Collector<OUT> out) throws Exception;
	    public abstract void processBroadcastElement(final IN2 value, final Context ctx, final Collector<OUT> out) throws Exception;
	}


### 案例二

如果Data Stream 是Non-Keyed Stream，则连接到Broadcasted Stream 后，添加处理ProcessFunction 时需要使用BroadcastProcessFunction 来实现， 下面是BroadcastProcessFunction 的API，代码如下所示：

	public abstract class BroadcastProcessFunction<IN1, IN2, OUT> extends BaseBroadcastProcessFunction {
			public abstract void processElement(final IN1 value, final ReadOnlyContext ctx, final Collector<OUT> out) throws Exception;
			public abstract void processBroadcastElement(final IN2 value, final Context ctx, final Collector<OUT> out) throws Exception;
	}
