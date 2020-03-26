# NIO

## 核心组件

Java NIO的核心组件 包括：

- 通道（Channel）
- 缓冲区（Buffer）
- 选择器（Selectors）


![](../Images/1.png)

NIO基于Channel和Buffer(缓冲区)进行操作，数据总是从通道读取到缓冲区中，或者从缓冲区写入到通道中。Selector(选择区)用于监听多个通道的事件（比如：连接打开，数据到达）。因此，单个线程可以监听多个数据通道。


## 具体使用


###  基于通道 & 缓冲数据

    // 1. 获取数据源 和 目标传输地的输入输出流（此处以数据源 = 文件为例）
    FileInputStream fin = new FileInputStream(infile);
    FileOutputStream fout = new FileOutputStream(outfile);

    // 2. 获取数据源的输入输出通道
    FileChannel fcin = fin.getChannel();
    FileChannel fcout = fout.getChannel();

    // 3. 创建 缓冲区 对象：Buffer（共有2种方法）
     // 方法1：使用allocate()静态方法
     ByteBuffer buff = ByteBuffer.allocate(256);
     // 上述方法创建1个容量为256字节的ByteBuffer
     // 注：若发现创建的缓冲区容量太小，则重新创建一个大小合适的缓冲区

    // 方法2：通过包装一个已有的数组来创建
     // 注：通过包装的方法创建的缓冲区保留了被包装数组内保存的数据
     ByteBuffer buff = ByteBuffer.wrap(byteArray);

     // 额外：若需将1个字符串存入ByteBuffer，则如下
     String sendString="你好,服务器. ";
     ByteBuffer sendBuff = ByteBuffer.wrap(sendString.getBytes("UTF-16"));

    // 4. 从通道读取数据 & 写入到缓冲区
    // 注：若 以读取到该通道数据的末尾，则返回-1
    fcin.read(buff);

    // 5. 传出数据准备：将缓存区的写模式 转换->> 读模式
    buff.flip();

    // 6. 从 Buffer 中读取数据 & 传出数据到通道
    fcout.write(buff);

    // 7. 重置缓冲区
    // 目的：重用现在的缓冲区,即 不必为了每次读写都创建新的缓冲区，在再次读取之前要重置缓冲区
    // 注：不会改变缓冲区的数据，只是重置缓冲区的主要索引值
    buff.clear();


### 基于选择器（Selecter）
	
	// 1. 创建Selector对象   
	Selector sel = Selector.open();
	
	// 2. 向Selector对象绑定通道   
	 // a. 创建可选择通道，并配置为非阻塞模式   
	 ServerSocketChannel server = ServerSocketChannel.open();   
	 server.configureBlocking(false);   
	 
	 // b. 绑定通道到指定端口   
	 ServerSocket socket = server.socket();   
	 InetSocketAddress address = new InetSocketAddress(port);   
	 socket.bind(address);   
	 
	 // c. 向Selector中注册感兴趣的事件   
	 server.register(sel, SelectionKey.OP_ACCEPT);    
	 return sel;
	
	// 3. 处理事件
	try {    
	    while(true) { 
	        // 该调用会阻塞，直到至少有一个事件就绪、准备发生 
	        selector.select(); 
	        // 一旦上述方法返回，线程就可以处理这些事件
	        Set<SelectionKey> keys = selector.selectedKeys(); 
	        Iterator<SelectionKey> iter = keys.iterator(); 
	        while (iter.hasNext()) { 
	            SelectionKey key = (SelectionKey) iter.next(); 
	            iter.remove(); 
	            process(key); 
	        }    
	    }    
	} catch (IOException e) {    
	    e.printStackTrace();   
	}



































































