# Netty三大组件之Buffer

## 基本介绍

缓冲区（Buffer）：缓冲区本质上是一个可以读写数据的内存块，可以理解成是一个容器对象(含数组)，该对象提供了一组方法，可以更轻松地使用内存块，，缓冲区对象内置了一些机制，能够跟踪和记录缓冲区的状态变化情况。Channel 提供从文件、网络读取数据的渠道，但是读取或写入的数据都必须经由 Buffer，如图:  

![](../Images/3.png)


## 代码Demo

	package com.wzy;
	
	import java.nio.IntBuffer;
	
	public class test {
	
	    public static void main(String[] args) {
	
	        //buffer数量为5
	        IntBuffer intBuffer = IntBuffer.allocate(5);
	
	        //往buffer里面写5个数字
	        for (int i=0;i<intBuffer.capacity();i++){
	            intBuffer.put(i*2);
	        }
	
	        //切换buffer（从写到读）
	        intBuffer.flip();
	
	        //读取buffer的内容
	        while (intBuffer.hasRemaining()){
	            System.out.println(intBuffer.get());
	        }
	    }
	}


## 打印结果

	0
	2
	4
	6
	8