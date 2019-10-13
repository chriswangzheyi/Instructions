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