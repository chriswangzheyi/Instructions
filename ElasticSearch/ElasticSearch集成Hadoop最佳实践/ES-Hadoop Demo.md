# ES-Hadoop Demo

## HdfsToES

	import org.apache.hadoop.conf.Configuration;
	import org.apache.hadoop.fs.Path;
	import org.apache.hadoop.io.BytesWritable;
	import org.apache.hadoop.io.NullWritable;
	import org.apache.hadoop.io.Text;
	import org.apache.hadoop.mapreduce.Job;
	import org.apache.hadoop.mapreduce.Mapper;
	import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
	import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
	import org.elasticsearch.hadoop.mr.EsOutputFormat;
	
	import java.io.IOException;
	
	
	public class HdfsToES {
	
	    public static class MyMapper extends Mapper<Object, Text, NullWritable,
	            Text> {
	        private Text line = new Text();
	        public void map(Object key, Text value, Mapper<Object, Text,
	                NullWritable, Text>.Context context) throws IOException, InterruptedException {
	
	            if(value.getLength()>0){
	                line.set(value);
	                context.write(NullWritable.get(), line);
	            }
	
	        }
	    }
	
	
	    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
	
	        System.out.println("-------------------------进来了------------------------");
	
	        Configuration conf = new Configuration();
	        conf.setBoolean("mapred.map.tasks.speculative.execution", false);
	        conf.setBoolean("mapred.reduce.tasks.speculative.execution", false);
	        conf.set("es.nodes", "47.112.142.231:9200");
	        conf.set("es.resource", "blog/csdn");
	        conf.set("es.mapping.id", "id");
	        conf.set("es.input.json", "yes");
	
	        Job job = Job.getInstance(conf, "hadoop es write test");
	        job.setMapperClass(HdfsToES.MyMapper.class);
	        job.setInputFormatClass(TextInputFormat.class);
	        job.setOutputFormatClass(EsOutputFormat.class);
	        job.setMapOutputKeyClass(NullWritable.class);
	        job.setMapOutputValueClass(BytesWritable.class);
	
	        // 设置输入路径
	        FileInputFormat.setInputPaths(job, new Path
	                ("hdfs://47.112.142.231:9000//input/blog.json"));
	        job.waitForCompletion(true);
	
	    }
	}



## POM

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0"
	         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	
	    <groupId>com.wzy</groupId>
	    <artifactId>es_hadoop_demo</artifactId>
	    <version>1.0-SNAPSHOT</version>
	
	    <properties>
	        <java.version>1.8</java.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.elasticsearch</groupId>
	            <artifactId>elasticsearch-hadoop</artifactId>
	            <version>7.6.2</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hadoop</groupId>
	            <artifactId>hadoop-common</artifactId>
	            <version>3.2.1</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hadoop</groupId>
	            <artifactId>hadoop-hdfs</artifactId>
	            <version>3.2.1</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hadoop</groupId>
	            <artifactId>hadoop-client</artifactId>
	            <version>3.2.1</version>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.httpcomponents</groupId>
	            <artifactId>httpclient</artifactId>
	            <version>4.5.3</version>
	        </dependency>
	        <dependency>
	            <groupId>commons-httpclient</groupId>
	            <artifactId>commons-httpclient</artifactId>
	            <version>3.1</version>
	        </dependency>
	    </dependencies>
	
	
	</project>