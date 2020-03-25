# AtomXXX系列

## AtomXXX类

该类封装在java.util.concurrent.atomic包中，包中包括了AtomicInteger，Atomicboolean，AtomicIntegerArray等等类型。 

这为我们提供了解决同样的加锁的问题的更高效的方法，使用AtomXXX类。AtomXXX类本身方法都是原子性的，但不能保证多个方法连续调用是原子性的。 

也就是说，单个的方法都是具有原子性的(例如自增)。但是，他们的间隙是可以插入的，也就是说，整体仍然是不具有原子性的，仍然需要加锁来保证整个代码块的原子性。 

例如，我们在实例化一个名为count的变量时：

	AtomicInteger count = new AtomicInteger(0);

对于常规的int类型，count++自然是不具有原子性的。由于JAVA没有符号重载，使用一个方法来实现原子自增：

	count.incrementAndGet(); //count++

**Atomic是基于unsafe类和自旋操作实现的，**


## atomicInteger Demo


	import java.util.concurrent.atomic.AtomicInteger;
	
	public class Test {
	
	
	    public static void main(String[] args) {
	        AtomicInteger atomicInteger=new AtomicInteger();
	
	        for(int i=0;i<10;i++){
	            Thread t=new Thread(new AtomicTest(atomicInteger));
	            t.start();
	            try {
	                t.join(0);
	            } catch (InterruptedException e) {
	                e.printStackTrace();
	            }
	        }
	
	        System.out.println(atomicInteger.get());
	    }
	}
	
	class AtomicTest implements Runnable{
	
	    AtomicInteger atomicInteger;
	
	    public AtomicTest(AtomicInteger atomicInteger){
	        this.atomicInteger=atomicInteger;
	    }
	
	    @Override
	    public void run() {
	        atomicInteger.addAndGet(1);
	        atomicInteger.addAndGet(2);
	        atomicInteger.addAndGet(3);
	        atomicInteger.addAndGet(4);
	    }
	
	}


最终的输出结果为100，可见这个程序是线程安全的。如果把AtomicInteger换成变量i的话，那最终结果就不确定了。

打开AtomicInteger的源码可以看到:

	// setup to use Unsafe.compareAndSwapInt for updates
	private static final Unsafe unsafe = Unsafe.getUnsafe();
	private volatile int value;


volatile关键字用来保证内存的可见性（但不能保证线程安全性），线程读的时候直接去主内存读，写操作完成的时候立即把数据刷新到主内存当中。