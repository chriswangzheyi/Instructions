# Window Function



## 类别

- ReduceFunction （Incremental ）
- AggregateFunction （ Incremental ）
- FoldFunction (Incremental)
- ProcessWindowFunction （All Element）





## ReduceFunction（增量）



`educeFunction` 是一种增量计算的窗口函数，用于组合窗口中的元素以生成单个结果值。每当窗口中接收到新的元素时，这个函数就会被调用，它将新元素与当前的累积结果进行合并，从而更新累积结果。这种方式效率较高，因为它仅存储并更新单个结果，而不是存储窗口内的所有元素。`ReduceFunction` 常用于实现如总和、最大值或最小值等简单聚合。



## AggregateFunction（增量）



`AggregateFunction` 类似于 `ReduceFunction`，但提供了更灵活的增量聚合功能。这个函数由三个部分组成：

- **创建累加器**：初始化一个用于存储中间聚合状态的累加器。
- **添加输入值到累加器**：定义如何将新的数据元素添加到累加器。
- **从累加器获取结果**：当需要输出结果时，定义如何从累加器计算最终结果。 `AggregateFunction` 通常用于更复杂的聚合，如计算平均值，其中需要同时跟踪总和和计数。

## FoldFunction（增量）



`FoldFunction` 是另一种用于聚合的增量函数，但在许多现代流处理框架中，它已经被 `AggregateFunction` 替代。`FoldFunction` 从一个初始值开始，并将每个输入元素折叠到当前的聚合结果中。虽然它类似于 `ReduceFunction`，但 `FoldFunction` 允许结果类型与输入类型不同。



## ProcessWindowFunction（全元素）

`ProcessWindowFunction` 处理整个窗口的所有元素，提供对窗口内所有数据的完全访问。这使得它能够实现比简单聚合更复杂的逻辑，如访问窗口的元数据（例如，窗口的开始和结束时间），或输出多个结果值。由于需要存储窗口中的所有元素，这种函数在内存使用上不如增量聚合函数高效。通常，`ProcessWindowFunction` 可以与其他增量聚合函数结合使用，以实现复杂的聚合逻辑，同时保持较高的效率。



## Demo





### 示例环境设置

```scala
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FoldFunction;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;

// 假设有一个温度传感器的数据源，数据类型为（timestamp, temperature）
DataStream<Tuple2<Long, Double>> sensorData = // 假设这里已经连接到了数据源

```





### 使用 ReduceFunction 计算最大温度

```scala
sensorData
    .keyBy(value -> value.f0) // 根据传感器ID分组
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .reduce(new ReduceFunction<Tuple2<Long, Double>>() {
        @Override
        public Tuple2<Long, Double> reduce(Tuple2<Long, Double> value1, Tuple2<Long, Double> value2) {
            return new Tuple2<>(value1.f0, Math.max(value1.f1, value2.f1));
        }
    })
    .print();

```



### 使用 AggregateFunction 计算平均温度

```scala
sensorData
    .keyBy(value -> value.f0)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new AggregateFunction<Tuple2<Long, Double>, Tuple2<Double, Integer>, Double>() {
        @Override
        public Tuple2<Double, Integer> createAccumulator() {
            return new Tuple2<>(0.0, 0);
        }

        @Override
        public Tuple2<Double, Integer> add(Tuple2<Long, Double> value, Tuple2<Double, Integer> accumulator) {
            return new Tuple2<>(accumulator.f0 + value.f1, accumulator.f1 + 1);
        }

        @Override
        public Double getResult(Tuple2<Double, Integer> accumulator) {
            return accumulator.f0 / accumulator.f1;
        }

        @Override
        public Tuple2<Double, Integer> merge(Tuple2<Double, Integer> a, Tuple2<Double, Integer> b) {
            return new Tuple2<>(a.f0 + b.f0, a.f1 + b.f1);
        }
    })
    .print();

```

