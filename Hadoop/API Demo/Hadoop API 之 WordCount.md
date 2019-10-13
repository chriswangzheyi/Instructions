# Hadoop API 之 WordCount

##准备测试文档

this is a test
just a test
Alice was beginning to get very tired of sitting by her sister on the bank
and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, `and what is the use of a book,' thought Alice `without pictures or conversation?' 
So she was considering in her own mind



## Maven项目


![](../Images/1.png)




**pom.xml**


	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	
	    <groupId>com.cwzy</groupId>
	    <artifactId>wordcount</artifactId>
	    <version>6.1.0-SNAPSHOT</version>
	    <packaging>jar</packaging>
	
	    <name>recommend</name>
	    <url>http://maven.apache.org</url>
	
	    <properties>
	        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>junit</groupId>
	            <artifactId>junit</artifactId>
	            <version>3.8.1</version>
	            <scope>test</scope>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hadoop</groupId>
	            <artifactId>hadoop-client</artifactId>
	            <version>2.2.0</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hadoop</groupId>
	            <artifactId>hadoop-common</artifactId>
	            <version>2.2.0</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hadoop</groupId>
	            <artifactId>hadoop-hdfs</artifactId>
	            <version>2.2.0</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hbase</groupId>
	            <artifactId>hbase-server</artifactId>
	            <version>0.98.3-hadoop2</version>
	            <exclusions>
	                <exclusion>
	                    <artifactId>hadoop-common</artifactId>
	                    <groupId>org.apache.hadoop</groupId>
	                </exclusion>
	            </exclusions>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hbase</groupId>
	            <artifactId>hbase-client</artifactId>
	            <version>0.98.3-hadoop2</version>
	            <exclusions>
	                <exclusion>
	                    <artifactId>hadoop-common</artifactId>
	                    <groupId>org.apache.hadoop</groupId>
	                </exclusion>
	            </exclusions>
	        </dependency>
	    </dependencies>
	</project>







**WordcountDriver：**

主类


	package com.wzy;
	
	//三步走：1.map
	//       2.reduce
	//       3.driver写驱动将以上两个类关联，运行
	
	import com.sun.istack.NotNull;
	import org.apache.hadoop.conf.Configuration;
	import org.apache.hadoop.fs.Path;
	import org.apache.hadoop.io.IntWritable;
	import org.apache.hadoop.io.Text;
	import org.apache.hadoop.mapreduce.Job;
	import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
	import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
	
	
	import java.io.IOException;
	
	public class WordcountDriver {
	    public static void main(@NotNull String[] args) throws IOException, ClassNotFoundException, InterruptedException {
	//　　　　　　　　　　　　　　此处标红的地方为在运行配置中所加入的参数，main里的参数很少用到，要用就要在运行时配置，如下图
	//        1.获取Job对象
	        Configuration conf =new Configuration();
	        Job job = Job.getInstance(conf);
	
	//        2.设置jar存储位置:有两种：
	//         setJar(String jar) : 参数即为要存储的 固定的  路径，不灵活，不能修改
	//         setJarByClass(Class<?> cls):Set the Jar by finding where a given class came from
	//                                    按类反射，灵活，可动态变化
	        job.setJarByClass(WordcountDriver.class);//存储位置根据WordcountDriver类的位置
	
	//        3.关联map和reduce类
	        job.setMapperClass(WordcountMapper.class);
	        job.setReducerClass(WordcountReducer.class);
	
	//        4.(固定格式)设置mapper阶段输出数据的key和value类型
	        job.setMapOutputKeyClass(Text.class);
	        job.setMapOutputValueClass(IntWritable.class);
	//        5.(固定格式)设置最终数据输出的key和value类型
	        job.setOutputKeyClass(Text.class);
	        job.setOutputValueClass(IntWritable.class);
	//        6.设置输入路径和输出路径
	        FileInputFormat.setInputPaths(job,new Path(args[0]));//args为main中的形参，输入路径设置为第一个参数，下图
	        FileOutputFormat.setOutputPath(job,new Path(args[1]));//输出路径设置为第二个参数，这种设置方式，可以便于打包后在集群上运行！
	
	        System.out.println("输出路径:"+args[0]+" ,输出路径: "+args[1]);
	//        7.提交Job
	//        job.submit();
	        boolean result = job.waitForCompletion(true);//设置为true，提交成功会打印相应信息
	    }
	}



**WordcountMapper：**

	package com.wzy;
	
	import org.apache.hadoop.io.IntWritable;
	import org.apache.hadoop.io.LongWritable;
	import org.apache.hadoop.io.Text;
	import org.apache.hadoop.mapreduce.Mapper;
	
	import java.io.IOException;
	//map阶段
	//Mapper<keyIn,value,keyOut,value>
	//keyOut输出key的类型 <mark，1>，<fun,1>等  按照方法说明来
	public class WordcountMapper extends Mapper<LongWritable, Text,Text, IntWritable> {
	    @Override
	    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
	        Text k = new Text();
	        IntWritable v = new IntWritable(1);//设置默认为1
	//        1.获取文件中的一行内容（默认都是这样）
	//          比如第一行为mark fun
	        String line = value.toString();
	//        2.切割单词
	        String[] words = line.split(" ");
	//        3.循环写出
	        for (String word: words) {
	//            写出的内容要按照上面KEYOU的类型，有Text和IntWritable
	
	            k.set(word);    //k为mark
	//            v.set(1);       //v为1
	            context.write(k,v);
	
	        }
	    }
	}


**WordcountReducer：**

	package com.wzy;
	
	import org.apache.hadoop.io.IntWritable;
	import org.apache.hadoop.io.Text;
	import org.apache.hadoop.mapreduce.Reducer;
	
	import java.io.IOException;
	
	import org.apache.hadoop.io.IntWritable;
	import org.apache.hadoop.io.Text;
	import org.apache.hadoop.mapreduce.Reducer;
	
	import java.io.IOException;
	
	//KEYIN,VALUEIN : 为map阶段输出的(k,v)
	//KEYOUT,VALUEOUT : 最终输出的(k,v)
	
	public class WordcountReducer extends Reducer<Text, IntWritable,Text,IntWritable> {
	    //   由于int与IntWritable（为一个类）类型不同，需要特别设置
	    IntWritable v = new IntWritable();
	
	    @Override
	    protected void reduce(Text key, Iterable<IntWritable> values, Context context)
	            throws IOException, InterruptedException {
	//        1.累加求和
	        int sum = 0;
	        for (IntWritable value:values) {
	            sum=sum+value.get();
	        }
	//        特别设置v
	        v.set(sum);//类型转换
	//        2.写出
	        context.write(key,v);
	    }
	}





## 部署


创建目录

	hadoop fs -mkdir /input3 

将测试文件传入文件夹中

	hadoop fs -put /root/test.txt /input3

验证是否上传成功

	hadoop fs -ls /input3 


将项目用Maven 打包

	mvn install

	#生成wordcount-6.1.0-SNAPSHOT.jar 

上传文件到服务器， 以/root路径为例子


运行程序 

	hadoop jar /root/wordcount-6.1.0-SNAPSHOT.jar com.wzy.WordcountDriver /input3 /output/1002

查看结果

	hadoop fs -cat /output/1002/part-r-00000