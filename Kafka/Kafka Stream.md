# Kafka Stream

参考：https://blog.csdn.net/weixin_48185778/article/details/111321994

## 核心概念

### KTable和KSteam


KTable和KSteam是Kafka中非常重要的概念，在此分析一下二者区别。

* KStream是一个数据流，可以认为所有的记录都通过Insert only的方式插入进这个数据流中。
* KTable代表一个完整的数据集，可以理解为数据库中的表。每条记录都是KV键值对，key可以理解为数据库中的主键，是唯一的，而value代表一条记录。我们可以认为KTable中的数据时通过Update only的方式进入的。如果是相同的key，会覆盖掉原来的那条记录。

综上来说，KStream是数据流，来多少数据就插入多少数据，是Insert only；KTable是数据集，相同key只允许保留最新的记录，也就是Update only

### 时间

####事件发生时间：

事件发生的时间，包含在数据记录中。发生时间由Producer在构造ProducerRecord时指定。并且需要Broker或者Topic将message.timestamp.type设置为CreateTime（默认值）才能生效。

####消息接收时间：

也即消息存入Broker的时间。当Broker或Topic将message.timestamp.type设置为LogAppendTime时生效。此时Broker会在接收到消息后，存入磁盘前，将其timestamp属性值设置为当前机器时间。一般消息接收时间比较接近于事件发生时间，部分场景下可代替事件发生时间。

####消息处理时间：

即Kafka Stream处理消息时的时间。


### 窗口

1）Hopping Time Window：举一个典型的应用场景，每隔5秒钟输出一次过去1个小时内网站的PV或者UV。里面有两个时间1小时和5秒钟，1小时指定了窗口的大小(Window size)，5秒钟定义输出的时间间隔(Advance interval)。

2）Tumbling Time Window：可以认为是Hopping Time Window的一种特例，窗口大小=输出时间间隔，它的特点是各个Window之间完全不相交。

3）Sliding Window该窗口只用于2个KStream进行Join计算时。该窗口的大小定义了Join两侧KStream的数据记录被认为在同一个窗口的最大时间差。假设该窗口的大小为5秒，则参与Join的2个KStream中，记录时间差小于5的记录被认为在同一个窗口中，可以进行Join计算。

4）Session Window该窗口用于对Key做Group后的聚合操作中。它需要对Key做分组，然后对组内的数据根据业务需求定义一个窗口的起始点和结束点。一个典型的案例是，希望通过Session Window计算某个用户访问网站的时间。对于一个特定的用户（用Key表示）而言，当发生登录操作时，该用户（Key）的窗口即开始，当发生退出操作或者超时时，该用户（Key）的窗口即结束。窗口结束时，可计算该用户的访问时间或者点击次数等。


## 应用示例

### pom依赖

	<dependency>
	  <groupId>org.apache.kafka</groupId>
	  <artifactId>kafka_2.11</artifactId>
	  <version>2.0.0</version>
	</dependency>
	<dependency>
	   <groupId>org.apache.kafka</groupId>
	   <artifactId>kafka-streams</artifactId>
	   <version>2.0.0</version>
	</dependency>


### 案例一：Kafka Stream的wordcount案例


	public class StreamSample {
	 
	    private static final String INPUT_TOPIC="stream-in";
	    private static final String OUT_TOPIC="stream-out";
	 
	    public static void main(String[] args) {
	        Properties props = new Properties();
	        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG,"localhost:9092");
	        props.put(StreamsConfig.APPLICATION_ID_CONFIG,"wordcount-app");
	        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
	        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());
	 
	        // 如果构建流结构拓扑
	        final StreamsBuilder builder = new StreamsBuilder();
	        // 构建Wordcount
	//        wordcountStream(builder);
	        // 构建foreachStream
	        foreachStream(builder);
	 
	        final KafkaStreams streams = new KafkaStreams(builder.build(), props);
	 
	        streams.start();
	    }
	 
	    // 如果定义流计算过程
	    static void foreachStream(final StreamsBuilder builder){
	        KStream<String,String> source = builder.stream(INPUT_TOPIC);
	        source
	                .flatMapValues(value -> Arrays.asList(value.toLowerCase(Locale.getDefault()).split(" ")))
	                .foreach((key,value)-> System.out.println(key + " : " + value));
	    }
	 
	    // 如果定义流计算过程
	    static void wordcountStream(final StreamsBuilder builder){
	        // 不断从INPUT_TOPIC上获取新数据，并且追加到流上的一个抽象对象
	        KStream<String,String> source = builder.stream(INPUT_TOPIC);
	        // Hello World imooc
	        // KTable是数据集合的抽象对象
	        // 算子
	        final KTable<String, Long> count =
	                source
	                        // flatMapValues -> 将一行数据拆分为多行数据  key 1 , value Hello World
	                        // flatMapValues -> 将一行数据拆分为多行数据  key 1 , value Hello key xx , value World
	                        /*
	                            key 1 , value Hello   -> Hello 1  World 2
	                            key 2 , value World
	                            key 3 , value World
	                         */
	                        .flatMapValues(value -> Arrays.asList(value.toLowerCase(Locale.getDefault()).split(" ")))
	                        // 合并 -> 按value值合并
	                        .groupBy((key, value) -> value)
	                        // 统计出现的总数
	                        .count();
	 
	        // 将结果输入到OUT_TOPIC中
	        count.toStream().to(OUT_TOPIC, Produced.with(Serdes.String(),Serdes.Long()));
	    }
	 
	}

### 案例二：将topicA的数据写入到topicB中(纯复制)

	import org.apache.kafka.common.serialization.Serdes;
	import org.apache.kafka.streams.KafkaStreams;
	import org.apache.kafka.streams.StreamsBuilder;
	import org.apache.kafka.streams.StreamsConfig;
	import org.apache.kafka.streams.Topology;
	
	import java.util.Properties;
	import java.util.concurrent.CountDownLatch;
	
	public class MyStream {
	    public static void main(String[] args) {
	        Properties prop =new Properties();
	        prop.put(StreamsConfig.APPLICATION_ID_CONFIG,"mystream");  
	        prop.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG,"192.168.136.20:9092"); //zookeeper的地址
	        prop.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass()); //输入key的类型
	        prop.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG,Serdes.String().getClass());  //输入value的类型
	
	        //创建流构造器
	        StreamsBuilder builder = new StreamsBuilder();
	
	        //构建好builder，将myStreamIn topic中的数据写入到myStreamOut topic中
	        builder.stream("myStreamIn").to("myStreamOut");
	
	        final Topology topo=builder.build();
	        final KafkaStreams streams = new KafkaStreams(topo, prop);
	
	        final CountDownLatch latch = new CountDownLatch(1);
	        Runtime.getRuntime().addShutdownHook(new Thread("stream"){
	            @Override
	            public void run() {
	                streams.close();
	                latch.countDown();
	            }
	        });
	        try {
	            streams.start();
	            latch.await();
	        } catch (InterruptedException e) {
	            e.printStackTrace();
	        }
	        System.exit(0);
	    }
	}
	
开启zookeeper和Kafka

	# 开启zookeeper
	zkServer.sh start
	
	# 后台启动Kafka
	kafka-server-start.sh -daemon /opt/kafka/config/server.properties

创建topic myStreamIn

	kafka-topics.sh --create --zookeeper 192.168.136.20:2181 --topic myStreamIn --partitions 1 --replication-factor 1	
创建topic myStreamOut

	kafka-topics.sh --create --zookeeper 192.168.136.20:2181 --topic myStreamOut --partitions 1 --replication-factor 1

生产消息写入到myStreamIn

	kafka-console-producer.sh --topic myStreamIn --broker-list 192.168.136.20:9092

消费myStreamOut里的数据

	kafka-console-consumer.sh --topic myStreamOut --bootstrap-server 192.168.136.20:9092 --from-beginning

运行示例代码并在生产者端输入数据，能在消费端看到数据，表明Kafka Stream写入成功。


