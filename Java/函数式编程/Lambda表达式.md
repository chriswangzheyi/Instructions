# Lambda表达式

## 定义

## 语法

	(parameters) -> expression
	或
	(parameters) ->{ statements; }

## 示例

![](../Images/1.png)

![](../Images/2.png)

### 静态方法引用例子
	
	public class test {
	
	    public static String staticTest(){
	        return "11";
	    }
	
	    public  static String staticTestParam(String input){
	        return input;
	    }
	
	    public static void main(String[] args) throws Exception {
	        result r1 = test::staticTest;
	        System.out.println("r1="+r1.getAns());
	
	        result2 r2 = test::staticTestParam;
	        System.out.println(r2.getValue("hh"));
	    }
	
	}
	
	interface  result {
	
	    String getAns();
	}
	
	interface  result2{
	    String getValue(String info);
	}


### 实例方法引用例子

	public class test {
	
	     void doPrint(){
	        System.out.println("111");
	    }
	
	     void doPrintPara(String info){
	        System.out.println(info);
	    }
	
	    public static void main(String[] args) {
	         new test().doPrint();
	         new test().doPrintPara("222");
	    }
	    
	}

### 对象方法的例子
	
	public class test {
	
	    public static void main(String[] args) {
	
	       test tt= new myTest().doStuff();
	
	    }
	
	}
	
	class myTest{
	
	    public test doStuff(){
	        System.out.println("hello");
	
	        return null;
	    }
	}


### 构造器方法的例子

	import java.util.LinkedHashSet;
	import java.util.Set;
	import java.util.function.Supplier;
	
	public class test {
	
	    public static void main(String[] args) {
	
	        Supplier<File> f = () ->new File("D:/1.txt");
	        System.out.println(f.get().canRead()); //f.get()
	
	        Supplier<Set> s = () -> new LinkedHashSet();
	        System.out.println(s.get());
	    }   
	}

## Stream API 

### 特性：

1： 不是数据结构、没有内部存储

2：不支持索引访问

3：延迟计算

4：支持并行

5:很容易生成数组和集合

6:支持过滤、查找、转换、汇总，聚合等操作

### 运行机制

Stream 分为源source，中间操作，终止操作

流的源可以使一个数组、一个集合、一个生成器方法、一个I/O通道

一个流可以有零个或者多个中间操作，每一个中间操作都会返回一个新的流，供一下一个操作使用。一个流只会有一个终止操作

Stream只有遇到终止操作，它的源才开始执行遍历操作

### 常见API

过滤 filter

    Arrays.asList(1,2,3,4,5).stream().filter(x -> x%2 ==0).forEach(System.out::println);

	#输出
	2
	4
	

去重 distinct

	Arrays.asList(1,1,2,2,3).stream().distinct().forEach(System.out::println);

	#输出
	1
	2
	3

排序 sorted

	Arrays.asList(1,3,2,4,5).stream().sorted().forEach(System.out::println);

	#输出
	1
	2
	3
	4
	5

截取 limit\skpi

	Arrays.asList(1,3,2,4,5).stream().limit(2).forEach(System.out::println);

	#输出
	1
	3

转换 map/flatMap

	 Arrays.asList(1,3,2,4,5).stream().map(Integer::doubleValue).forEach(System.out::println);

	#输出
	1.0
	3.0
	2.0
	4.0
	5.0

其他 peek


### 终止操作

循环 forEach

	#例子
	Arrays.asList(1,2,3,4,5).stream().forEach(num -> System.out.println(num+1));

	#输出
	2
	3
	4
	5
	6

计算 min、 max 、 count、 average

	#例子
	double value = Arrays.asList(1,3,2,4,5).stream().max(Integer::compareTo).get();
	System.out.println(value);

	#输出
	5

匹配 anyMatch、 allMatch、 noneMatch、findFirst、findAny

	#例子
	boolean value = Arrays.asList(1,3,2,4,5).stream().anyMatch(num ->num>3);
	
	#输出
	System.out.println(value);

汇聚 reduce

![](../Images/3.png)

	#例子
	int ans = Arrays.asList(1,2,3,4,5).stream().reduce(0,(a,b) ->  a + b);
        System.out.println(ans);

	#输出
	15

收集器 toArray collect

	#例子
	List<String> list= Arrays.asList("a", "b", "c", "d");
	
	List<String> collect =list.stream().map(String::toUpperCase).collect(Collectors.toList());
	System.out.println(collect);

	#输出
	[A, B, C, D]


### Stream的创建

通过数组

	String[] arr = {"a","b","1","2"};
	Stream<String> myStream = Stream.of(arr);

通过集合

	List<String> mylist = Arrays.asList("a","b","c","d");
	Stream<List> myStream = Stream.of(mylist);

通过Stream.generate方法

    static  void gen3(){
        Stream<Integer> myGenerator = Stream.generate(()->1);
    }


通过Stream.iterate方法

    static  void gen4(){
        Stream<Integer> myGenerator = Stream.iterate(1, x->x+1);
    }


通过其他API


## Stream 例子

### 将一个数组改类型并且每个数字加1

1,2,3,4 -> 2.0, 3.0, 4.0, 5.0

	import java.util.Arrays;
	import java.util.List;
	import java.util.stream.Collectors;
	
	public class test {
	
	    public static void main(String[] args) {
	
	        List<Integer> list= Arrays.asList(1,2,3,4);
	        List<Double> ans = list.stream().map(Integer::doubleValue).map(num -> num+1).collect(Collectors.toList());
	        System.out.println(ans);
	
	    }
	}


### 将一个数组的数字求和

1,2,3,4 -> 10

	List<Integer> list= Arrays.asList(1,2,3,4);
	Optional<Integer> ans = list.stream().reduce((a,b) ->a+b);
	System.out.println(ans.get());


## 串行流

单线程顺序运行。可以用sequential()或者不写来表示

    Optional<Integer> max = Stream.iterate(1,x->x+1).limit(200).peek(x ->{System.out.println(Thread.currentThread().getName());}).max(Integer::compare);
    System.out.println(max);

	#输出
	main
	main
	main
	main

## 并行流

把一个内容分成多个数据块，并用不同的线程分别处理每个数据块的流。使用方法：在Stream上加一个parallel方法


    Optional<Integer> max = Stream.iterate(1,x->x+1).limit(200).parallel().peek(x ->{System.out.println(Thread.currentThread().getName());}).max(Integer::compare);
    System.out.println(max);

	#输出
	ForkJoinPool.commonPool-worker-7
	ForkJoinPool.commonPool-worker-4
	ForkJoinPool.commonPool-worker-7
	main
	main
	main
	main
	ForkJoinPool.commonPool-worker-5
	ForkJoinPool.commonPool-worker-5
	ForkJoinPool.commonPool-worker-2
	ForkJoinPool.commonPool-worker-2
	ForkJoinPool.commonPool-worker-1

## Fork/Join框架

### Fork/Join框架与传统线程池的区别

![](../Images/4.png)


## 实战

### 切割参数

"itemId=1&userId=10000&type=20&token=12312312&key=123123" 切割为map

	import java.util.Map;
	import java.util.stream.Collectors;
	import java.util.stream.Stream;
	
	
	public class demo {
	    
	    @Test
	    public void test1(){
	        String query = "itemId=1&userId=10000&type=20&token=12312312&key=123123";
	
	        //toMap后参数是key : value。 数组的第一个值是key，第二个是value
	        Map<String, String> param = Stream.of(query.split("&")).map(str -> str.split("=")).collect( Collectors.toMap(s ->s[0], s -> s[1]) );
	        System.out.println(param);
	    }
	    
}


分析：

Stream.of(query.split("&"))： 以&为界，分割参数

map(str -> str.split("="))：以=为界，分割参数，并生成一个数组

collect( Collectors.toMap(s ->s[0], s -> s[1]) ： 将数组的第一个数据作为key, 数组的第二个数据作为value


