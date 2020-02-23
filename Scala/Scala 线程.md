# Scala 线程

参考：https://blog.csdn.net/weixin_40873462/article/details/89680070

## 通过扩展Thread类

以下示例扩展了Thread类并覆盖了run方法，start()方法用于启动线程。

	class ThreadDemo extends Thread{
	  override def run(): Unit = {
	    println("Thread is running")
	  }
	}
	
	object  Demo{
	  def main(args: Array[String]): Unit = {
	    var t =new ThreadDemo()
	    t.start();
	  }
	}

## 通过扩展Runnable接口

	class ThreadDemo extends Runnable{
	  override def run(): Unit = {
	    println("thread is running")
	  }
	}
	
	object  Demo{
	  def main(args: Array[String]): Unit = {
	
	    var run= new ThreadDemo
	    var t = new Thread(run)
	    t.start()
	
	  }
	}
