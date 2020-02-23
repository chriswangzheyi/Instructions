# Scala 统计词频Demo


## 代码

	object wordCount {
	
	  def main(args: Array[String]): Unit = {
	
	    val lines = List("hadoop hive spark scala", "spark hive habase", "hive spark java")
	    //数据转换
	    val mappedWords = lines.flatMap(_.split(" ").map(_.trim)).filterNot(_.isEmpty).map((_, 1))
	    println("---------------输出分割后的值-----------------")
	    println(mappedWords)
	    //根据数据进行分组
	    val groupedWords: Map[String, List[(String, Int)]] = mappedWords.groupBy(tuple => tuple._1)
	    //每组进行数据计算
	    println("---------------输出分组后的值-----------------")
	    println(groupedWords)
	
	    val result = groupedWords.map(tuple => {
	      //获得Word单词（key）
	      val word = tuple._1
	      //计算该Word对应的数量（value）
	      val count = tuple._2.map(t => t._2).sum
	      //返回结果
	      (word, count)
	    })
	    println("---------------输出合并后的值-----------------")
	    println(result)
	    println("---------------转换成list的值-----------------")
	    println(result.toList)
	    
	  }
	
	}


## 代码分析


### lines.flatMap(_.split(" ").map(_.trim))

将原始数据去除空格后以" "为分隔符分开，并返回所有的元素

输出：

List(hadoop, hive, spark, scala, spark, hive, habase, hive, spark, java)

### map((_, 1))

将所有元素，以键值对的形式返回。元素为key，1为值

输出：

List((hadoop,1), (hive,1), (spark,1), (scala,1), (spark,1), (hive,1), (habase,1), (hive,1), (spark,1), (java,1))


### mappedWords.groupBy(tuple => tuple._1)

tuple => tuple._1 ： 输入元组，返回元组的第一个元素。

比如(hadoop,1)， 返回的即为hadoop

输出：

Map(java -> List((java,1)), hadoop -> List((hadoop,1)), spark -> List((spark,1), (spark,1), (spark,1)), hive -> List((hive,1), (hive,1), (hive,1)), scala -> List((scala,1)), habase -> List((habase,1)))


###     val result = groupedWords.map(tuple =>

将上面的输出作为一个元组输入，将list的key做为word，将list的value求和。 并返回结果