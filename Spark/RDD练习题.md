## RDD练习题


## 练习1 

val rdd1 = sc.parallelize(List(5, 6, 4, 7, 3, 8, 2, 9, 1, 10))，对RDD1中的每个元素乘以2，然后排序,并过滤到大于10的值
		
	scala> val rdd1 = sc.parallelize(List(5, 6, 4, 7, 3, 8, 2, 9, 1, 10))
	rdd1: org.apache.spark.rdd.RDD[Int] = ParallelCollectionRDD[12] at parallelize at <console>:24		  
	
	scala> val rdd2 = rdd1.map(_*2).sortBy(x=>x ,true).filter(_ > 10).collect
	rdd2: Array[Int] = Array(12, 14, 16, 18, 20)
		 
##  练习2 
val rdd1 = sc.parallelize(Array("a b c", "d e f", "h i j"))  对每个元素先切分再压平


	scala> val rdd1 = sc.parallelize(Array("a b c", "d e f", "h i j"))  
	rdd1: org.apache.spark.rdd.RDD[String] = ParallelCollectionRDD[21] at parallelize at <console>:24

	scala> rdd1.flatMap(_.split(" ")).collect
	res13: Array[String] = Array(a, b, c, d, e, f, h, i, j)


## 练习3 

val rdd1 = sc.parallelize(List(5, 6, 4, 3))

val rdd2 = sc.parallelize(List(1, 2, 3, 4))

### 求并集
	
	scala> rdd1.union(rdd2).collect
	res17: Array[Int] = Array(5, 6, 4, 3, 1, 2, 3, 4)
	
### 交集

	scala> rdd1.intersection(rdd2).collect
	res16: Array[Int] = Array(4, 3)
	
### 去重
	rdd3.distinct.collect
	rdd4.collect


##  练习4
 
	val rdd1 = sc.parallelize(List(("tom", 1), ("jerry", 3), ("kitty", 2)))
	val rdd2 = sc.parallelize(List(("jerry", 2), ("tom", 1), ("shuke", 2)))
	
### 执行链接操作：join

	scala> val rdd3 = rdd1.join(rdd2).collect
	rdd3: Array[(String, (Int, Int))] = Array((tom,(1,1)), (jerry,(3,2)))
	
### 并集

	scala> val rdd4 = rdd1 union rdd2
	rdd4: org.apache.spark.rdd.RDD[(String, Int)] = UnionRDD[5] at union at <console>:27
	
	scala> rdd4.groupByKey.collect
	res0: Array[(String, Iterable[Int])] = Array((tom,CompactBuffer(1, 1)), (jerry,CompactBuffer(3, 2)), (shuke,CompactBuffer(2)), (kitty,CompactBuffer(2)))
	
	

## 练习5 

	val rdd1 = sc.parallelize(List(1, 2, 3, 4, 5))
	
聚合操作

	scala> rdd1.reduce(_+_)
	res4: Int = 15
	      
	      
	      

## 练习6 

	val rdd1 = sc.parallelize(List(("tom", 1), ("jerry", 3), ("kitty", 2), ("shuke", 1)))
	val rdd2 = sc.parallelize(List(("jerry", 2), ("tom", 3), ("shuke", 2), ("kitty", 5)))
	val rdd3 = rdd1.union(rdd2)
	
### 按key进行聚合操作

	scala> val rdd4 = rdd3.reduceByKey(_+_)
	rdd4: org.apache.spark.rdd.RDD[(String, Int)] = ShuffledRDD[11] at reduceByKey at <console>:25
	
	scala> rdd4.collect
	res5: Array[(String, Int)] = Array((tom,4), (jerry,5), (shuke,3), (kitty,7))

