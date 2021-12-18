# Spark的缓存机制

## 调用方法

	persist和cache ----> 是一种Transformation
	

## 优点

Spark RDD中间不产生临时数据，单分布式系统风险很高，所以容易出错。出错以后如果有持久化，就可以根据血统重新计算处理。如果没有持久化则需要从头计算。

## 什么时候需要持久化

* 某个步骤非常耗时
* 计算链条非常长，重新计算需要很多步骤	
* checkpoint所在RDD需要
* Shuffle之后需要persist，因为shuffle要进行网络传输，风险很大
* shuffle之前进行persist，框架默认将数据持久化到磁盘
	
	
## 缓存位置

		  val NONE = new StorageLevel(false, false, false, false)
		  val DISK_ONLY = new StorageLevel(true, false, false, false)
		  val DISK_ONLY_2 = new StorageLevel(true, false, false, false, 2)
		  val MEMORY_ONLY = new StorageLevel(false, true, false, true)
		  val MEMORY_ONLY_2 = new StorageLevel(false, true, false, true, 2)
		  val MEMORY_ONLY_SER = new StorageLevel(false, true, false, false)
		  val MEMORY_ONLY_SER_2 = new StorageLevel(false, true, false, false, 2)
		  val MEMORY_AND_DISK = new StorageLevel(true, true, false, true)
		  val MEMORY_AND_DISK_2 = new StorageLevel(true, true, false, true, 2)
		  val MEMORY_AND_DISK_SER = new StorageLevel(true, true, false, false)
		  val MEMORY_AND_DISK_SER_2 = new StorageLevel(true, true, false, false, 2)
		  val OFF_HEAP = new StorageLevel(true, true, true, false, 1)
		  
带数字表示副本数量

MEMORY_ONLY如果内存放不下，则		  
		  
## Demo步骤

		(1) 测试数据放到HDFS:    /data/sales
		(2) 使用RDD读取文件
		     val rdd1 = sc.textFile("hdfs://hadoop111:9000/data/sales")
			 
		(3) 求个数   rdd1.count   -----> 没有缓存，耗费时间：  3s
		(4) 缓存数据: rdd1.cache    rdd1.persist  -----> 是一种Transformation，延时加载
		    生效：    rdd1.count   -----> 没有缓存，耗费时间：  2s
			
		(5) 再执行一次： rdd1.count -----> 有缓存，耗费时间：0.1 s

由上面的例子可以看出，缓存后，速度变快


