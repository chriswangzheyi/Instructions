# Volatile 

## 含义

一旦一个共享变量（类的成员变量、类的静态成员变量）被 volatile 修饰之后，那么就具备了两层语义：

1）保证了不同线程对这个变量进行操作时的可见性，即一个线程修改了某个变量的值，这新值对其他线程来说是立即可见的。

2）禁止进行指令重排序。

**但是不能保证操作的原子性**

## 可见性概念例子


### 不用volatile关键字修饰的时候

	public class volatileDemo implements Runnable{
	
	    private static boolean flag = false;
	
	    public void run() {
	        while (!flag) ;
	    }
	
	    public static void main(String[] args) throws Exception {
	
	        Thread thread1 = new Thread(new volatileDemo());
	
	        thread1.start();
	
	        thread1.sleep(2000);
	
	        flag=true;
	    }
	
	}


main线程会一直执行，但是thread1会一直卡住。 因为主线程执行到flag = ture的时候，并不会通知thread1线程flag值的改变。 


### 用volatile关键字修饰的时候

	public class volatileDemo implements Runnable{
	
	    private static volatile boolean flag = false;
	
	    public void run() {
	        while (!flag) ;
	    }
	
	    public static void main(String[] args) throws Exception {
	
	        Thread thread1 = new Thread(new volatileDemo());
	
	        thread1.start();
	
	        thread1.sleep(2000);
	
	        flag=true;
	    }
	
	}

main线程和thread1线程都会运行后退出，因为main线程会通知thread1，flag的变化。



##总结

对于volatile变量的读/写操作是原子性的。因为从内存屏障的角度来看，对volatile变量的单纯读写操作确实没有任何疑问。

由于其中掺杂了一个自增的CPU内部操作，就造成这个复合操作不再保有原子性。

然后，讨论了如何保证volatile++这类操作的原子性，比如使用synchronized或者AtomicInteger/AtomicLong原子类
