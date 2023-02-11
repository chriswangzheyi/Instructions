# Spark 窗口函数
参考：https://blog.csdn.net/bitcarmanlee/article/details/113617901


## 构造数据集

	import org.apache.spark.SparkConf
	import org.apache.spark.sql.{Row, SparkSession}
	import org.apache.spark.sql.functions._
	
	  def test() = {
	    val sparkConf = new SparkConf().setMaster("local[2]")
	    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
	
	    val data = Array(("lili", "ml", 90),
	      ("lucy", "ml", 85),
	      ("cherry", "ml", 80),
	      ("terry", "ml", 85),
	      ("tracy", "cs", 82),
	      ("tony", "cs", 86),
	      ("tom", "cs", 75))
	
	    val schemas = Seq("name", "subject", "score")
	    val df = spark.createDataFrame(data).toDF(schemas: _*)
	
	    df.show()
	 }

### 打印

	+------+-------+-----+
	|  name|subject|score|
	+------+-------+-----+
	|  lili|     ml|   90|
	|  lucy|     ml|   85|
	|cherry|     ml|   80|
	| terry|     ml|   85|
	| tracy|     cs|   82|
	|  tony|     cs|   86|
	|   tom|     cs|   75|
	+------+-------+-----+
	
## 分组查看排名

### 三种窗口函数

	  def test() = {
	    val sparkConf = new SparkConf().setMaster("local[2]")
	    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
	    val sqlContext = spark.sqlContext
	
	
	    val data = Array(("lili", "ml", 90),
	      ("lucy", "ml", 85),
	      ("cherry", "ml", 80),
	      ("terry", "ml", 85),
	      ("tracy", "cs", 82),
	      ("tony", "cs", 86),
	      ("tom", "cs", 75))
	
	    val schemas = Seq("name", "subject", "score")
	    val df = spark.createDataFrame(data).toDF(schemas: _*)
	    df.createOrReplaceTempView("person_subject_score")
	
	    val sqltext = "select name, subject, score, rank() over (partition by subject order by score desc) as rank from person_subject_score";
	    val ret = sqlContext.sql(sqltext)
	    ret.show()
	
	    val sqltext2 = "select name, subject, score, row_number() over (partition by subject order by score desc) as row_number from person_subject_score";
	    val ret2 = sqlContext.sql(sqltext2)
	    ret2.show()
	
	    val sqltext3 = "select name, subject, score, dense_rank() over (partition by subject order by score desc) as dense_rank from person_subject_score";
	    val ret3 = sqlContext.sql(sqltext3)
	    ret3.show()
	  }

### 打印

	+------+-------+-----+----+
	|  name|subject|score|rank|
	+------+-------+-----+----+
	|  tony|     cs|   86|   1|
	| tracy|     cs|   82|   2|
	|   tom|     cs|   75|   3|
	|  lili|     ml|   90|   1|
	|  lucy|     ml|   85|   2|
	| terry|     ml|   85|   2|
	|cherry|     ml|   80|   4|
	+------+-------+-----+----+
	
	+------+-------+-----+----------+
	|  name|subject|score|row_number|
	+------+-------+-----+----------+
	|  tony|     cs|   86|         1|
	| tracy|     cs|   82|         2|
	|   tom|     cs|   75|         3|
	|  lili|     ml|   90|         1|
	|  lucy|     ml|   85|         2|
	| terry|     ml|   85|         3|
	|cherry|     ml|   80|         4|
	+------+-------+-----+----------+
	
	+------+-------+-----+----------+
	|  name|subject|score|dense_rank|
	+------+-------+-----+----------+
	|  tony|     cs|   86|         1|
	| tracy|     cs|   82|         2|
	|   tom|     cs|   75|         3|
	|  lili|     ml|   90|         1|
	|  lucy|     ml|   85|         2|
	| terry|     ml|   85|         2|
	|cherry|     ml|   80|         3|
	+------+-------+-----+----------+
