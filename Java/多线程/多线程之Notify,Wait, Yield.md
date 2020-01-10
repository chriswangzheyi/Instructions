# 多线程之Notify,Wait, Yield

## Notify Wait

Notify:唤醒一个被wait阻塞的线程

NotifyAll： 唤醒全部被wait阻塞的线程

Wait: 阻塞当前线程，直到被唤醒

### Demo

	public class myTest {
	
	    static final Object obj = new Object();
	
	
	    public static void main(String[] args) throws InterruptedException {
	
	        Thread thread1 = new Thread(new worker1(), "worker1 thread");
	        Thread thread2 = new Thread(new worker2(), "worker2 thread");
	        Thread thread3 = new Thread(new worker3(), "worker3 thread");
	
	        thread2.start();
	        thread3.start();
	
	        thread2.join(1000);
	        thread3.join(1000);
	
	        thread1.start();
	    }
	
	
	     static class worker1 implements  Runnable{
	
	        @Override
	        public void run() {
	
	            synchronized (obj){
	                System.out.println("1111111111 进来了");
	
	                System.out.println("1111111111 开始唤醒");
	                obj.notifyAll();
	
	                System.out.println("1111111111 完成了");
	            }

	        }
	    }
	
	
	     static class worker2 implements Runnable{
	
	        @Override
	        public void run() {
	
	            synchronized (obj){
	                System.out.println("22222222222222进来了");
	
	                try {
	                    System.out.println("22222222222222 开始等待");
	                    obj.wait();
	                } catch (InterruptedException e) {
	                    e.printStackTrace();
	                }
	
	                System.out.println("22222222222222 完成了");
	            }
	
	        }
	    }
	
	
	    static class worker3 implements Runnable{
	
	        @Override
	        public void run() {
	
	            synchronized (obj){
	                System.out.println("333333333333进来了");
	
	                try {
	                    System.out.println("333333333333 开始等待");
	                    obj.wait();
	                } catch (InterruptedException e) {
	                    e.printStackTrace();
	                }
	
	                System.out.println("333333333333 完成了");
	            }
	
	        }
	    }
	
	}





### 测试结果


	22222222222222进来了
	22222222222222 开始等待
	333333333333进来了
	333333333333 开始等待
	1111111111 进来了
	1111111111 开始唤醒
	1111111111 完成了
	333333333333 完成了
	22222222222222 完成了



## yield(), sleep(), join() 区别


sleep执行后线程进入阻塞状态

yield执行后线程进入就绪状态

join执行后线程进入阻塞状态



### sleep()：

使当前线程（即调用该方法的线程）暂停执行一段时间，让其他线程有机会继续执行，但它并不释放对象锁。也就是说如果有synchronized同步快，其他线程仍然不能访问共享数据。注意该方法要捕捉异常。


### join()：

join()方法使调用该方法的线程在此之前执行完毕，也就是等待该方法的线程执行完毕后再往下继续执行。注意该方法也需要捕捉异常。


### yield()：

该方法与sleep()类似，只是不能由用户指定暂停多长时间，并且yield（）方法只能让同优先级的线程有执行的机会。