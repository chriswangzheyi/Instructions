# Spark的缓存机制

## 调用方法

	persist和cache ----> 是一种Transformation
	
	
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
		  
## Demo步骤

		(1) 测试数据放到HDFS:    /data/sales
		(2) 使用RDD读取文件
		     val rdd1 = sc.textFile("hdfs://hadoop111:9000/data/sales")
			 
		(3) 求个数   rdd1.count   -----> 没有缓存，耗费时间：  3s
		(4) 缓存数据: rdd1.cache    rdd1.persist  -----> 是一种Transformation，延时加载
		    生效：    rdd1.count   -----> 没有缓存，耗费时间：  2s
			
		(5) 再执行一次： rdd1.count -----> 有缓存，耗费时间：0.1 s

由上面的例子可以看出，缓存后，速度变快