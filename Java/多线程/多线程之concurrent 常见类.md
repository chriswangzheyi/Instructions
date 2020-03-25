# concurrent 常见类

---


## executor 


## Lock


## atomic


## Callable

## ConcurrentHashMap

线程安全的Hashmap


## ReentrantLock 

参考资料：https://www.cnblogs.com/takumicx/p/9338983.html

重入锁

jdk中独占锁的实现除了使用关键字synchronized外,还可以使用ReentrantLock。虽然在性能上ReentrantLock和synchronized没有什么区别，但ReentrantLock相比synchronized而言功能更加丰富，使用起来更为灵活，也更适合复杂的并发场景。


### Demo

	import java.util.concurrent.locks.ReentrantLock;
	
	public class Test {
	
	    public static void main(String[] args) throws InterruptedException {
	
	        ReentrantLock lock = new ReentrantLock();
	
	        for (int i = 1; i <= 3; i++) {
	            lock.lock();
	        }
	
	        for(int i=1;i<=3;i++){
	            try {
	
	            } finally {
	                lock.unlock();
	            }
	        }
	    }
	
	}


上面的代码通过lock()方法先获取锁三次，然后通过unlock()方法释放锁3次，程序可以正常退出。从上面的例子可以看出,ReentrantLock是可以重入的锁,当一个线程获取锁时,还可以接着重复获取多次。