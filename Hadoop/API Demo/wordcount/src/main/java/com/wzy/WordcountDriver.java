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