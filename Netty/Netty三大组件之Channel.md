# Netty三大组件之Channel


## 基本介绍

(1) NIO的通道类似于流，但有些区别如下：

1. 通道可以同时进行读写，而流只能读或者只能写
1. 通道可以实现异步读写数据
1. 通道可以从缓冲读数据，也可以写数据到缓冲: 

(2) BIO 中的 stream 是单向的，例如 FileInputStream 对象只能进行读取数据的操作，而 NIO 中的通道(Channel)是双向的，可以读操作，也可以写操作。

(3)Channel在NIO中是一个接口public interface Channel extends Closeable{} 

(4)常用的 Channel 类有：FileChannel、DatagramChannel、ServerSocketChannel 和 SocketChannel。【ServerSocketChanne 类似 ServerSocket , SocketChannel 类似 Socket】

(5)FileChannel 用于文件的数据读写，DatagramChannel 用于 UDP 的数据读写，ServerSocketChannel 和 SocketChannel 用于 TCP 的数据读写


## Demo1  往本地写文件

	package com.wzy;
	
	import java.io.FileOutputStream;
	import java.io.IOException;
	import java.nio.ByteBuffer;
	import java.nio.channels.FileChannel;
	
	public class test {
	
	    public static void main(String[] args) throws IOException {
	
	        String str = "test";
	        //创建一个输出流
	        FileOutputStream fileOutputStream = new FileOutputStream("D:\\1.txt");
	
	        //通过fileoutputstream获取对应的filechannel
	        FileChannel fileChannel = fileOutputStream.getChannel();
	
	        //创建一个缓冲区
	        ByteBuffer byteBuffer =ByteBuffer.allocate(1024);
	
	        //将str 放入到buyebuffer中
	        byteBuffer.put(str.getBytes());
	
	        //对bytebuffer进行翻转
	        byteBuffer.flip();
	
	        //将缓冲区数据写入channel
	        fileChannel.write(byteBuffer);
	
	        //关闭流
	        fileOutputStream.close();
	    }
	}



运行后，在对应位置生成一个文件


## Demo2  从本地读文件

	package com.wzy;
	
	
	import java.io.File;
	import java.io.FileInputStream;
	import java.io.IOException;
	import java.nio.ByteBuffer;
	import java.nio.channels.FileChannel;
	
	public class test {
	
	    public static void main(String[] args) throws IOException {
	
	        //创建文件的输入流
	        File file = new File("D:\\1.txt");
	        FileInputStream fileInputStream = new FileInputStream(file);
	
	        //通过fileinputstream 获取对应的filechannel
	        FileChannel fileChannel = fileInputStream.getChannel();
	
	        //创建缓冲区
	        ByteBuffer byteBuffer = ByteBuffer.allocate( (int)file.length());
	
	        //将通道的数据读入buffer
	        fileChannel.read(byteBuffer);
	
	        //将字节数据转成String
	        System.out.println(new String(byteBuffer.array()));
			fileInputStream.close();
	
	    }
	}


## ## Demo3  从本地读写文件（只用一个Buffer）

	package com.wzy;
	
	import java.io.FileInputStream;
	import java.io.FileOutputStream;
	import java.io.IOException;
	import java.nio.ByteBuffer;
	import java.nio.channels.FileChannel;
	
	public class test {
	
	    public static void main(String[] args) throws IOException {
	
	        FileInputStream fileInputStream = new FileInputStream("1.txt");
	        FileChannel fileChannel01 = fileInputStream.getChannel();
	
	        FileOutputStream fileOutputStream =new FileOutputStream("2.txt");
	        FileChannel fileChannel02 = fileOutputStream.getChannel();
	
	        ByteBuffer byteBuffer = ByteBuffer.allocate(512);
	
	        while (true){
	
	            byteBuffer.clear();//将标志位重置(非常重要)
	            int read = fileChannel01.read(byteBuffer);
	            if (read == -1){//读取结束
	                break;
	            }
	            byteBuffer.flip();
	            fileChannel02.write(byteBuffer);
	        }
	        fileInputStream.close();
	        fileOutputStream.close();
	    }
	}

自建一个1.txt文件放在项目父目录即可测试。

## Demo4 拷贝图片

	package com.wzy;
	
	import java.io.FileInputStream;
	import java.io.FileOutputStream;
	import java.io.IOException;
	import java.nio.channels.FileChannel;
	
	public class test {
	
	    public static void main(String[] args) throws IOException {
	
	        FileInputStream fileInputStream = new FileInputStream("d:\\a.jpg");
	        FileChannel sourceCh = fileInputStream.getChannel();
	
	        FileOutputStream fileOutputStream =new FileOutputStream("d:\\a2.jpg");
	        FileChannel destCh = fileOutputStream.getChannel();
	
	        //使用transform完成拷贝
	        destCh.transferFrom(sourceCh,0,sourceCh.size());
	
	        //关闭流和通道
	        sourceCh.close();
	        destCh.close();
	        fileInputStream.close();
	        fileOutputStream.close();
	    }
	}
