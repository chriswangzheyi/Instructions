package com.wzy

import org.apache.spark.{SparkConf, SparkContext}

object WordCount {
  def main(args: Array[String]): Unit = {
    //设置分布式的运行平台，和appname
    //使用Master运行平台，yarn，standalong（spark自带的运行平台），mesos，local四种
    //local开发调试时用的环境，前三种一般为上线的运行环境
    //local local[N] local[*]
    val conf = new SparkConf().setMaster("yarn").setAppName("WordCount")
    //构建sparkContext对象
    val sc = new SparkContext(conf)
    //加载数据源，获取RDD对象
    val textFile = sc.textFile("hdfs://Master001:9000/input2/test1.txt")
    val counts = textFile.flatMap(line => line.split(" ")).map(word => (word,1)).reduceByKey(_+_)
    counts.saveAsTextFile("/spark/output/20191101")
  }
}
