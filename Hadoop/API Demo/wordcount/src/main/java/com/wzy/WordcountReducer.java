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