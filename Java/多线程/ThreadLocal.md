# ThreadLocal


## 定义

ThreadLocal类并不是用来解决多线程环境下的共享变量问题，而是用来提供线程内部的共享变量，在多线程环境下，可以保证各个线程之间的变量互相隔离、相互独立。


## 例子


	public class ThreadLocalTest {
	    static class MyThread extends Thread {
	        private static ThreadLocal<Integer> threadLocal = new ThreadLocal<>();
	
	        @Override
	        public void run() {
	            super.run();
	            for (int i = 0; i < 3; i++) {
	                threadLocal.set(i);
	                System.out.println(getName() + " threadLocal.get() = " + threadLocal.get());
	            }
	        }
	    }
	
	    public static void main(String[] args) {
	        MyThread myThreadA = new MyThread();
	        myThreadA.setName("ThreadA");
	
	        MyThread myThreadB = new MyThread();
	        myThreadB.setName("ThreadB");
	
	        myThreadA.start();
	        myThreadB.start();
	    }
	}


输出：

	ThreadA threadLocal.get() = 0
	ThreadB threadLocal.get() = 0
	ThreadA threadLocal.get() = 1
	ThreadB threadLocal.get() = 1
	ThreadB threadLocal.get() = 2
	ThreadA threadLocal.get() = 2