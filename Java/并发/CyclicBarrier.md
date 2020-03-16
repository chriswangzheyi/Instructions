# CyclicBarrier 

## 定义

它的作用就是会让所有线程都等待完成后才会继续下一步行动。可以重复使用。

## 重要源码分析

### 构造方法：

	public CyclicBarrier(int parties)
	public CyclicBarrier(int parties, Runnable barrierAction)

parties 是参与线程的个数

第二个构造方法有一个 Runnable 参数，这个参数的意思是最后一个到达线程要做的任务

### 重要方法

	public int await() throws InterruptedException, BrokenBarrierException
	public int await(long timeout, TimeUnit unit) throws InterruptedException, BrokenBarrierException, TimeoutException


线程调用 await() 表示自己已经到达栅栏

BrokenBarrierException 表示栅栏已经被破坏，破坏的原因可能是其中一个线程 await() 时被中断或者超时

## Demo

	import java.util.concurrent.CyclicBarrier;
	
	public class Test {
	
	    static class TaskThread extends Thread {
	
	        CyclicBarrier barrier;
	
	        public TaskThread(CyclicBarrier barrier) {
	            this.barrier = barrier;
	        }
	
	        @Override
	        public void run() {
	            try {
	                Thread.sleep(1000);
	                System.out.println(getName() + " 到达栅栏 A");
	                barrier.await();
	                System.out.println(getName() + " 冲破栅栏 A");
	
	                Thread.sleep(2000);
	                System.out.println(getName() + " 到达栅栏 B");
	                barrier.await();
	                System.out.println(getName() + " 冲破栅栏 B");
	            } catch (Exception e) {
	                e.printStackTrace();
	            }
	        }
	    }
	
	    public static void main(String[] args) {
	        int threadNum = 5;
	        CyclicBarrier barrier = new CyclicBarrier(threadNum, new Runnable() {
	
	            @Override
	            public void run() {
	                System.out.println(Thread.currentThread().getName() + " 完成最后任务");
	            }
	        });
	
	        for(int i = 0; i < threadNum; i++) {
	            new TaskThread(barrier).start();
	        }
	    }
	
	}


打印：
	
	Thread-0 到达栅栏 A
	Thread-1 到达栅栏 A
	Thread-3 到达栅栏 A
	Thread-4 到达栅栏 A
	Thread-2 到达栅栏 A
	Thread-2 完成最后任务
	Thread-2 冲破栅栏 A
	Thread-0 冲破栅栏 A
	Thread-4 冲破栅栏 A
	Thread-1 冲破栅栏 A
	Thread-3 冲破栅栏 A
	Thread-0 到达栅栏 B
	Thread-4 到达栅栏 B
	Thread-1 到达栅栏 B
	Thread-2 到达栅栏 B
	Thread-3 到达栅栏 B
	Thread-3 完成最后任务
	Thread-0 冲破栅栏 B
	Thread-4 冲破栅栏 B
	Thread-3 冲破栅栏 B
	Thread-2 冲破栅栏 B
	Thread-1 冲破栅栏 B


从打印结果可以看出，所有线程会等待全部线程到达栅栏之后才会继续执行，并且最后到达的线程会完成 Runnable 的任务。

## CyclicBarrier 使用场景

可以用于多线程计算数据，最后合并计算结果的场景。