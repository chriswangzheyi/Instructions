# Join 

## 定义 

作用：使当前线程交换执行权，等待另一个线程执行完毕后，继续执行。等待该线程终止。

目的：是为了将并行的执行变成串行的执行。

场景：线程 thread1 和 线程 thread2 同是执行，但是 线程 thread2 调用了join 方法后，就进入等待状态 wait。直到 thread1 执行完毕后执行。



## 例子

	public class JoinDemo {
	
	    private static int num =0;
	
	    public static void main(String[] args) throws InterruptedException {
	
	        Thread thread1 = new Thread(new Runnable() {
	            @Override
	            public void run() {
	
	                while (true){
	
	                    try {
	                        Thread.sleep(2000);
	                    } catch (InterruptedException e) {
	                        e.printStackTrace();
	                    }
	
	                    num++;
	                    System.out.println("线程1在工作");
	
	                    if (num>=5){break;}
	
	                }
	
	            }
	        });
	
	
	        Thread thread2 = new Thread(new Runnable() {
	            @Override
	            public void run() {
	                
	                    System.out.println("线程2在工作");
	
	            }
	        });
	
	
	        thread1.start();
	        thread1.join();
	        thread2.start();
	
	    }
	
	}


例子中，线程1执行完毕后才会执行线程2。