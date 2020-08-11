# Spark 常见算子

官方文档上列举共有32种常见算子，包括Transformation的20种操作和Action的12种操作。

## Transformation

### Map

map的输入变换函数应用于RDD中所有元素，而mapPartitions应用于所有分区。区别于mapPartitions主要在于调用粒度不同。如parallelize（1 to 10， 3），map函数执行10次，而mapPartitions函数执行3次。

### Filter

过滤操作，满足filter内function函数为true的RDD内所有元素组成一个新的数据集。如：filter（a == 1）。

### FlatMap（function）

map是对RDD中元素逐一进行函数操作映射为另外一个RDD，而flatMap操作是将函数应用于RDD之中的每一个元素，将返回的迭代器的所有内容构成新的RDD。而flatMap操作是将函数应用于RDD中每一个元素，将返回的迭代器的所有内容构成RDD。

flatMap与map区别在于map为“映射”，而flatMap“先映射，后扁平化”，map对每一次（func）都产生一个元素，返回一个对象，而flatMap多一步就是将所有对象合并为一个对象。


### mapPartitions（function）

区于foreachPartition（属于Action，且无返回值），而mapPartitions可获取返回值。与map的区别前面已经提到过了，但由于单独运行于RDD的每个分区上（block），所以在一个类型为T的RDD上运行时，（function）必须是Iterator<T> => Iterator 类型的方法（入参）。


## Action

### 

