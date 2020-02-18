# Scala基础知识


## 常量变量

val 常量

var 变量

## 类型

###scala是类型推断的

val x =10  自动赋予x int类型

也可以自定义类型

val x:int = 10


### Trait

类似Java的Interface

#### 例子

##### USB

	trait USB {
	  def slot();
	}


##### Mp3
	
	object Mp3 extends  USB {
	  override def slot(): Unit = ???
	}


可以通过extends去实现trait。

如果需要实现多个trait，则使用with 关键字

extends trait1 with trait2 with trait3


### 构造器

主构造器的参数直接放置在类名后面

Scala代码：

	class Person(val name:String, val age:Int)

(1) var 只生成get、set方法

(2) val 只对应get方法

(3)缺失var和val，如果不参与计算，则不会生成任何成员

(4) 缺失var和val，如果参与计算，升格为字段


对比Java 代码：

	public class Person{
	
		private String name;
		private int age;
		public Person(String name, int age){
			this.name = name;
			this.age = age;
		}
	
		public String name(){
			return this.name;
		}
	
		public int age(){
			return this.age;
		}
	
	}

### 单例对象

Scala不能定义静态成员，而是代之定义单例对象(singleton object)。以object关键字定义。对象定义了某个类的单个实例，包含了你想要的特性

	object Accounts{
	    private var lastNumber = 0
	    def newUniqueNumber() = { lastNumber += 1; lastNumber}
	}

当你在应用程序中需要一个新的唯一账号时，调用Account.newUniqueNumber()即可。对象的构造器在该对象第一次被使用时调用。

### 伴生对象

当单例对象与某个类共享同一个名称时，它就被称为是这个类的伴生对象(companion object)。类和它的伴生对象必须定义在同一个源文件中。类被称为是这个单例对象的伴生类(companion class)。类和它的伴生对象可以互相访问其私有成员

	class Account {
	    val id = Account.newUniqueNumber()
	    private var balance = 0.0
	    def deposit(amount: Double){ balance += amount }
	    ...
	}
	
	object Account { //伴生对象
	        private var lastNumber = 0
	        def newUniqueNumber() = { lastNumber += 1; lastNumber}
	}

### Class和Object的区别

类class里无static类型，类里的属性和方法，必须通过new出来的对象来调用，所以有main主函数也没用。

而object的特点是：

可以拥有属性和方法，且默认都是"static"类型，可以直接用object名直接调用属性和方法，不需要通过new出来的对象（也不支持）。

object里的main函数式应用程序的入口。

object和class有很多和class相同的地方，可以extends父类或Trait，但object不可以extends object，即object无法作为父类。



### 拉链操作

#### zip

将两个元组合并在一起。如果元组的长度不同，则以数量少的元组为准

	object test {
	
	  def main(args: Array[String]): Unit = {
	    
	    val year = List(2010, 2010, 2010, 2016, 2016)
	    val month = List(1, 2, 3, 9, 10)
	    val price = List(13.8, 32, 62.9, 66, 88, 99)
	    val profit = List(1.1, 2.2, 3.3, 4.4, 5.5)
	
	    val res1 = year zip month
	    println(res1)
	
	    val res2 = price zip profit
	    println(res2)
	
	  }
	}


输出结果
	
	List((2010,1), (2010,2), (2010,3), (2016,9), (2016,10))
	List((13.8,1.1), (32.0,2.2), (62.9,3.3), (66.0,4.4), (88.0,5.5))

#### zipAll

a.zipAll(b, thisElem, thatElem),a短，用thisElem填补；b短，用thatElem填补

	object test {
	
	  def main(args: Array[String]): Unit = {
	
	    val spark1 = List("Vector", "Feature")
	    val spark2 = List("Scala", "SQL", "MLlib", "GraphX", "Streaming")
	    val spark3 = List("111")
	
	    val ans1 = spark1 zip spark2;
	
	    /// a.zipAll(b, thisElem, thatElem),a短，用thisElem填补；b短，用thatElem填补
	    val ans2 = spark1.zipAll(spark2, "DataFrame", "Pipeline")
	
	    val ans3 = spark2.zipAll(spark3,"one","two");
	
	    println(ans1)
	    println(ans2)
	    println(ans3)
	
	  }
	}

输出结果

	List((Vector,Scala), (Feature,SQL))
	List((Vector,Scala), (Feature,SQL), (DataFrame,MLlib), (DataFrame,GraphX), (DataFrame,Streaming))
	List((Scala,111), (SQL,two), (MLlib,two), (GraphX,two), (Streaming,two))


#### zipWithIndex


将元组加索引，然后组成一个对偶


	object test {
	
	  def main(args: Array[String]): Unit = {
	
	    val spark2 = List("Scala", "SQL", "MLlib", "GraphX", "Streaming")
	    val spark3 = List("111")
	
	    val ans3 = spark2.zipAll(spark3,"one","two");
	
	    println(ans3)
	    println(ans3.zipWithIndex)
	
	  }
	}

输出结果

	List((Scala,111), (SQL,two), (MLlib,two), (GraphX,two), (Streaming,two))
	List(((Scala,111),0), ((SQL,two),1), ((MLlib,two),2), ((GraphX,two),3), ((Streaming,two),4))


### Nothing，Null，None，Nil

#### Nothing

Nothing是所有类型的子类，也是Null的子类。Nothing没有对象，但是可以用来定义类型。例如，如果一个方法抛出异常，则异常的返回值类型就是Nothing(虽然不会返回) 。


##### Null

Null是所有AnyRef的子类，在scala的类型系统中，AnyRef是Any的子类，同时Any子类的还有AnyVal。对应java值类型的所有类型都是AnyVal的子类。所以Null可以赋值给所有的引用类型(AnyRef)，不能赋值给值类型，这个java的语义是相同的。 null是Null的唯一对象。


#### None

one是一个object，是Option的子类型。} scala推荐在可能返回空的方法使用Option[X]作为返回类型。如果有值就返回Some[x] (Some也是Option的子类)，否则返回None

	def get(key: A): Option[B] = {
	    if (contains(key))
	        Some(getValue(key))
	    else
	        None
	}


#### Nil

Nil是一个空的List，定义为List[Nothing]，根据List的定义List[+A]，所有Nil是所有List[T]的子类。 



### 文件的读取

使用Source类完成


	import java.io.FileWriter
	import scala.io.Source
	
	object test {
	
	  def main(args: Array[String]): Unit = {
	
	    // 从文件读取
	    val source1 = Source.fromFile("D:/11.txt")
	    for (line <- source1.getLines()){
	        System.out.println(line)
	    }
	    source1.close();
	
	    //网络资源读取
	    val webFile=Source.fromURL("http://spark.apache.org")
	    webFile.foreach(print)
	    webFile.close()
	
	    // 写文件
	    val source2 = new FileWriter("D:/11.txt",true);
	    for(i <- 1 to 10){
	      source2.write("hello"+i+"\n")
	    }
	
	    source2.close();
	    
	  }
	}


## flatMap 与 filter 

flatMap 将集合中的每个元素的子元素映射到某个函数并返回新的集合。

filter：将符合要求的数据(筛选)放置到新的集合。


### filter

	object testFilter {
	
	  def main(args: Array[String]): Unit = {
	    val names = List("About", "Box", "Clear")
	
	    println(names.filter(startA))
	  }
	
	  def startA(str: String): Boolean = {
	    str.startsWith("A")
	  }
	}


输出

	List(About)

### flatMap


	object testFlatMap {
	
	  def main(args: Array[String]): Unit = {
	    val names = List("About", "Box", "Clear")
	
	    println(names.map(upper))
	    println(names.flatMap(upper))
	  }
	
	  def upper(s: String): String = {
	    s.toUpperCase()
	  }
	}


输出

	List(ABOUT, BOX, CLEAR)
	List(A, B, O, U, T, B, O, X, C, L, E, A, R)