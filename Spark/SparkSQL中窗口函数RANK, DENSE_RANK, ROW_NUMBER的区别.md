# SparkSQL中窗口函数RANK, DENSE_RANK, ROW_NUMBER的区别


## 概念

* RANK：可以生成不连续的序号，比如按分数排序，第一第二都是100分，第三名98分，那第一第二就会显示序号1，第三名显示序号3。
* DENSE_RANK： 生成连续的序号，在上一例子中，第一第二并列显示序号1，第三名会显示序号2。
* ROW_NUMBER: 顾名思义就是行的数值，在上一例子中，第一第二第三将会显示序号为1,2,3。




|  姓名   | 年级  | 分数  | RANK  | DENSE_RANK  | ROW_NUMBER |
|  ----  | ----  |----  |---  |---  |--- |--- |
| 张三  | 一年级 | 100 | 1| 1 | 1 |
| 李四  | 一年级 | 100 |1 | 1 | 2 |
| 王五  | 一年级 |98 |3 |2 |3 |
| 小明  | 二年级 | 100 |1 |1|1 |
| 小芳  | 二年级 |95 |2 |2 |2 |
| 小民  | 二年级 |90 |3 |3 |3 |


## 代码

	package SparkSQLScala
	 
	import org.apache.spark.sql.SparkSession
	import org.apache.spark.sql.expressions.Window
	import org.apache.spark.sql.functions._
	 
	object TestWindowFunction {
	  def main(args: Array[String]): Unit = {
	    val sparkSession = SparkSession.builder().master("local").appName("TestWindownFunction").getOrCreate()
	    sparkSession.sparkContext.setLogLevel("ERROR")
	//  创建测试数据
	    val ScoreDetailDF = sparkSession.createDataFrame(Seq(
	      ("王五", "一年级", 98),
	      ("李四", "一年级", 100),
	      ("小民", "二年级", 90),
	      ("小明", "二年级", 100),
	      ("张三", "一年级", 100),
	      ("小芳", "二年级", 95)
	    )).toDF("name", "grade", "score")
	 
	    //    SparkSQL 方法实现
	    ScoreDetailDF.createOrReplaceTempView("ScoreDetail")
	    sparkSession.sql("SELECT * , " +
	      "RANK() OVER (PARTITION BY grade ORDER BY score DESC) AS rank, " +
	      "DENSE_RANK() OVER (PARTITION BY grade ORDER BY score DESC) AS dense_rank, " +
	      "ROW_NUMBER() OVER (PARTITION BY grade ORDER BY score DESC) AS row_number " +
	      "FROM ScoreDetail").show()
	 
	    //    DataFrame API 实现
	    val windowSpec = Window.partitionBy("grade").orderBy(col("score").desc)
	    ScoreDetailDF.select(col("name"),
	      col("grade"),
	      col("score"),
	      rank().over(windowSpec).as("rank"),
	      dense_rank().over(windowSpec).as("dense_rank"),
	      row_number().over(windowSpec).as("row_number")
	    ).show()
	 
	  }
	 
	}

