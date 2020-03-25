# concurrentLinkedQueue

## 介绍

concurrentLinkedQueue是一个基于链接节点的无界线程安全队列。此队列按照 FIFO（先进先出）原则对元素进行排序。队列的头部 是队列中时间最长的元素。队列的尾部 是队列中时间最短的元素。
新的元素插入到队列的尾部，队列获取操作从队列头部获得元素。当多个线程共享访问一个公共 collection 时，ConcurrentLinkedQueue 是一个恰当的选择。此队列不允许使用 null 元素。


## 常见方法

	import java.util.concurrent.ConcurrentLinkedQueue;
	
	public class ConCurrentLinkedQueueDemo {
	
	    public static void main(String[] args) {
	
	        ConcurrentLinkedQueue queue = new ConcurrentLinkedQueue();
	
	        //插入队列尾部
	        queue.offer("first");
	
	        //移除队列头部
	        queue.poll();
	
	        //获取但不移除此队列的头；如果此队列为空，则返回 null
	        queue.peek();
	
	        //  从队列中移除指定元素的单个实例（如果存在）
	        queue.remove("first");
	
	        //如果此队列包含指定元素，则返回 true
	        queue.contains("first");
	    }
	}


