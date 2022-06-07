# Flink多流合并

## 1.Union算子的使用

返回值：DataStream->DataStream

功能：合并两个或多个数据流，创建包含所有流中的所有元素的新流。注意:如果你将一个数据流和它本身联合起来，你将在结果流中得到每个元素两次。也就是说Union操作是不会去重的。

另外，被union的两个流的数据类型必须要一致。

说明：通过union算子可以把两个数据流进行合并，这样当有多个具有相同类型的数据流时，就可以当（合并）成一个流来进行处理。从而让流的处理更加灵活。

	public class UionDemo1 {
	    public static void main(String[] args) throws Exception {
	        final StreamExecutionEnvironment senv =
	                StreamExecutionEnvironment.getExecutionEnvironment();
	
	        DataStream<Tuple2<String, Integer>> src1 = senv.fromElements(
	                new Tuple2<>("shanghai", 15),
	                new Tuple2<>("beijing", 25));
	
	        DataStream<Tuple2<String, Integer>> src2 = senv.fromElements(
	                new Tuple2<>("sichuan", 35),
	                new Tuple2<>("chongqing", 45));
	
	        DataStream<Tuple2<String, Integer>> src3 = senv.fromElements(
	                new Tuple2<>("shenzheng", 55),
	                new Tuple2<>("guanzhou", 65));
					
	      	// 这个流不能和以上几个流进行union，由于流数据的类型不同
					// DataStream<Integer> src4 = senv.fromElements(2, 3);
	      
	        DataStream<Tuple2<String, Integer>> union = src1.union(src2, src3);
	
	        union.filter(t->t.f1>30).print("union");
	        senv.execute();
	    }
	}

以上代码把3个流合并成一个流，然后对合并的流进行处理（过滤）。最后打印输出。输出的内容如下：

	union:5> (sichuan,35)
	union:6> (chongqing,45)
	union:6> (shenzheng,55)
	union:7> (guanzhou,65)


## connect算子的使用

返回值：DataStream,DataStream → ConnectedStream

功能：“连接” 两个数据流并保留各自的类型。connect 允许在两个流的处理逻辑之间共享状态。

Connect算子和Union算子的区别：

* （1）Connect算子可以合并两个不同类型的数据流，而Uion只能合并相同类型的数据流。
* （2）Connect算子只支持两个数据流的合并，union可以支持多个数据流的合并。
* （3）两个DataStream经过connect之后被转化为ConnectedStreams，ConnectedStreams会对两个流的数据应用不同的处理方法，且双流之间可以共享状态。

输入输出：ConnectedStream → DataStream
功能：类似于在连接的数据流上进行 map 和 flatMap


	public class ConnectOpDemo1 {
	    public static void main(String[] args) throws Exception {
	        final StreamExecutionEnvironment senv =
	                StreamExecutionEnvironment.getExecutionEnvironment();
	
	        DataStream<Tuple2<String, Integer>> src1 = senv.fromElements(
	                new Tuple2<>("shanghai", 15),
	                new Tuple2<>("beijing", 25));
	        
	        DataStream<Integer> src4 = senv.fromElements(2, 3);
	
	        ConnectedStreams<Tuple2<String, Integer>, Integer> connStream = src1.connect(src4);
	
	        // 对不同类型的流，进行不同的处理，并统一输出成一个新的数据类型。
	        // 这里，我把两个流的数据都转成了String类型，这样方便后续的处理。
	        DataStream<String> res = connStream.flatMap(new CoFlatMapFunction<Tuple2<String, Integer>, Integer, String>() {
	            @Override
	            public void flatMap1(Tuple2<String, Integer> value, Collector<String> out) {
	                out.collect(value.toString());
	            }
	
	            @Override
	            public void flatMap2(Integer value, Collector<String> out) {
	                String word = String.valueOf(value);
	                out.collect(word);
	            }
	        });
	
	        res.print();
	        senv.execute();
    }
