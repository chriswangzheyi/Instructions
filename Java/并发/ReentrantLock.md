# ReentrantLock

参考资料：https://www.cnblogs.com/takumicx/p/9338983.html

## 概念

jdk中独占锁的实现除了使用关键字synchronized外,还可以使用ReentrantLock。虽然在性能上ReentrantLock和synchronized没有什么区别，但ReentrantLock相比synchronized而言功能更加丰富，使用起来更为灵活，也更适合复杂的并发场景。


### ReentrantLock是独占锁且可重入的

	import java.util.concurrent.locks.ReentrantLock;
	
	public class ReentrantLockDemo {
	
	    public static void main(String[] args) {
	        ReentrantLock lock = new ReentrantLock();
	
	        for (int i = 1; i <= 3; i++) {
	            lock.lock();
	        }
	
	        //unlock的次数决定了子线程是否能运行，i=2时候无法运行，i=3的时候可以运行
	        for(int i=1;i<=2;i++){
	            try {
	
	            } finally {
	                lock.unlock();
	            }
	        }
	
	        Thread thread1 = new Thread(new Runnable() {
	            public void run() {
	                System.out.println("11111111111");
	            }
	        });
	
	        if (!lock.isLocked()) {
	            thread1.start();
	        }
	    }
	}



上面的代码通过lock()方法先获取锁三次，然后通过unlock()方法释放锁3次，程序可以正常退出。从上面的例子可以看出,ReentrantLock是可以重入的锁,当一个线程获取锁时,还可以接着重复获取多次。

1.ReentrantLock和synchronized都是独占锁,只允许线程互斥的访问临界区。但是实现上两者不同:synchronized加锁解锁的过程是隐式的,用户不用手动操作,优点是操作简单，但显得不够灵活。一般并发场景使用synchronized的就够了；ReentrantLock需要手动加锁和解锁,且解锁的操作尽量要放在finally代码块中,保证线程正确释放锁。ReentrantLock操作较为复杂，但是因为可以手动控制加锁和解锁过程,在复杂的并发场景中能派上用场。

2.ReentrantLock和synchronized都是可重入的。synchronized因为可重入因此可以放在被递归执行的方法上,且不用担心线程最后能否正确释放锁；而ReentrantLock在重入时要却确保重复获取锁的次数必须和重复释放锁的次数一样，否则可能导致其他线程无法获得该锁。



## 公平锁

公平锁是指当锁可用时,在锁上等待时间最长的线程将获得锁的使用权。而非公平锁则随机分配这种使用权。和synchronized一样，默认的ReentrantLock实现是非公平锁,因为相比公平锁，非公平锁性能更好。当然公平锁能防止饥饿,某些情况下也很有用。在创建ReentrantLock的时候通过传进参数true创建公平锁,如果传入的是false或没传参数则创建的是非公平锁

	ReentrantLock lock = new ReentrantLock(true);

公平锁的例子：

	import java.util.concurrent.TimeUnit;
	import java.util.concurrent.locks.Lock;
	import java.util.concurrent.locks.ReentrantLock;
	
	public class ReentrantLockTest {
	
	    static Lock lock = new ReentrantLock(true);
	
	    public static void main(String[] args) throws InterruptedException {
	
	        for(int i=0;i<5;i++){
	            new Thread(new ThreadDemo(i)).start();
	        }
	
	    }
	
	    static class ThreadDemo implements Runnable {
	        Integer id;
	
	        public ThreadDemo(Integer id) {
	            this.id = id;
	        }
	
	        @Override
	        public void run() {
	            try {
	                TimeUnit.MILLISECONDS.sleep(10);
	            } catch (InterruptedException e) {
	                e.printStackTrace();
	            }
	            for(int i=0;i<2;i++){
	                lock.lock();
	                System.out.println("获得锁的线程："+id);
	                lock.unlock();
	            }
	        }
	    }
	}


![](../Images/1.png)

可以看到，根据等待时间，各线程轮番获取了锁。


## 非公平锁

将ReentrantLock参数改为false即可

	static Lock lock = new ReentrantLock(false); 