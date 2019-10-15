package com.wzy;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class ChoosecolumnMapper extends Mapper<Object, Text,Text, Text> {

    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        super.map(key, value, context);

        String[] q = value.toString().split(" ");
        context.write(new Text(q[0]), new Text(""));
    }
}
