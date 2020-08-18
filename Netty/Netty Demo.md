# Netty Demo

echo 服务

## 项目结构
![](../Images/1.png)


## 源码

### EchoClient

	package com.wzy.echo;
	
	import io.netty.bootstrap.Bootstrap;
	import io.netty.channel.ChannelFuture;
	import io.netty.channel.ChannelInitializer;
	import io.netty.channel.EventLoopGroup;
	import io.netty.channel.nio.NioEventLoopGroup;
	import io.netty.channel.socket.SocketChannel;
	import io.netty.channel.socket.nio.NioSocketChannel;
	
	import java.net.InetSocketAddress;
	
	public class EchoClient {
	
	    private String host;
	    private int port;
	
	    public EchoClient(String host, int port) {
	        this.host = host;
	        this.port = port;
	    }
	
	    public void start(){
	        EventLoopGroup group = new NioEventLoopGroup();
	
	        Bootstrap bootstrap = new Bootstrap();
	        try {
	            bootstrap.group(group)
	                    .channel(NioSocketChannel.class)
	                    .remoteAddress(new InetSocketAddress(host,port))
	                    .handler(new ChannelInitializer<SocketChannel>() {
	                        protected void initChannel(SocketChannel socketChannel) throws Exception {
	                            socketChannel.pipeline().addLast(new EchoClientHandler());
	                        }
	                    });
	
	            //连接到服务器，connect是异步连接，再调用同步等待sync，等待连接成功
	            ChannelFuture channelFuture = bootstrap.connect().sync();
	
	            //阻塞知道客户端通道关闭
	            channelFuture.channel().closeFuture().sync();
	
	        } catch (InterruptedException e) {
	            e.printStackTrace();
	        } finally {
	            //释放NIO线程
	            group.shutdownGracefully();
	        }
	    }
	
	    public static void main(String[] args) {
	        new EchoClient("127.0.0.1",1234).start();
	    }
	}

### EchoClientHandler

	package com.wzy.echo;
	
	import io.netty.buffer.ByteBuf;
	import io.netty.buffer.Unpooled;
	import io.netty.channel.ChannelHandlerContext;
	import io.netty.channel.SimpleChannelInboundHandler;
	import io.netty.util.CharsetUtil;
	
	public class EchoClientHandler extends SimpleChannelInboundHandler<ByteBuf> {
	
	    protected void channelRead0(ChannelHandlerContext channelHandlerContext, ByteBuf msg) throws Exception {
	
	        System.out.println("Client snet :" + msg.toString(CharsetUtil.UTF_8) );
	
	    }
	
	    @Override
	    public void channelActive(ChannelHandlerContext ctx) throws Exception {
	        System.out.println("Clent Actived");
	        ctx.writeAndFlush(Unpooled.copiedBuffer("wangzheyi is testing!",CharsetUtil.UTF_8));
	    }
	
	    @Override
	    public void channelReadComplete(ChannelHandlerContext ctx) throws Exception {
	        System.out.println("Client Completed");
	    }
	
	    @Override
	    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) throws Exception {
	        cause.printStackTrace();
	        ctx.close();
	    }
	
	}


### EchoServer

	package com.wzy.echo;
	
	import io.netty.bootstrap.ServerBootstrap;
	import io.netty.channel.ChannelFuture;
	import io.netty.channel.ChannelInitializer;
	import io.netty.channel.EventLoopGroup;
	import io.netty.channel.nio.NioEventLoopGroup;
	import io.netty.channel.socket.SocketChannel;
	import io.netty.channel.socket.nio.NioServerSocketChannel;
	
	public class EchoServer {
	
	    private int port;
	
	    public EchoServer(int port) {
	        this.port = port;
	    }
	
	    public  void run() throws InterruptedException {
	        //配置服务端线程组
	        EventLoopGroup bossGroup = new NioEventLoopGroup();
	        EventLoopGroup workGroup = new NioEventLoopGroup();
	
	        try {
	            ServerBootstrap serverBootstrap = new ServerBootstrap();
	            serverBootstrap.group(bossGroup,workGroup)
	                    .channel(NioServerSocketChannel.class)
	                    .childHandler(new ChannelInitializer<SocketChannel>() {
	                        protected void initChannel(SocketChannel socketChannel) throws Exception {
	                            //流水线中增加作业内容(handler)
	                            socketChannel.pipeline().addLast(new EchoServerHandler());
	                        }
	                    });
	            System.out.println("Echo服务器启动");
	
	            //绑定端口,同步等待绑定成功
	            ChannelFuture channelFuture = serverBootstrap.bind(port).sync();
	
	            //等待监听端口关闭
	            channelFuture.channel().closeFuture().sync();
	        } finally {
	            //优雅退出，释放线程池
	            workGroup.shutdownGracefully();
	            bossGroup.shutdownGracefully();
	        }
	    }
	
	    public static void main(String[] args) throws InterruptedException {
	        int port = 1234;
	
	        if (args.length >0){
	            port = Integer.parseInt(args[0]);
	        }
	
	        new EchoServer(port).run();
	    }
	    
	}


### EchoServerHandler

	package com.wzy.echo;
	
	import io.netty.buffer.ByteBuf;
	import io.netty.channel.ChannelHandlerContext;
	import io.netty.channel.ChannelInboundHandlerAdapter;
	import io.netty.util.CharsetUtil;
	
	public class EchoServerHandler extends ChannelInboundHandlerAdapter {
	
	    @Override
	    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
	
	        //接收信息
	        ByteBuf data = (ByteBuf) msg;
	        System.out.println("服务端收到数据："+data.toString(CharsetUtil.UTF_8));
	
	        // 回写
	        ctx.writeAndFlush(data);
	    }
	
	    @Override
	    public void channelReadComplete(ChannelHandlerContext ctx) throws Exception {
	        System.out.println("读取完毕");
	    }
	
	    @Override
	    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) throws Exception {
	        cause.printStackTrace();
	        ctx.close();
	    }
	}


### pom.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0"
	         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	
	    <groupId>com.wzy</groupId>
	    <artifactId>netty_test</artifactId>
	    <version>1.0-SNAPSHOT</version>
	
	    <dependencies>
	        <!-- netty-all 依赖 -->
	        <dependency>
	            <groupId>io.netty</groupId>
	            <artifactId>netty-all</artifactId>
	            <version>4.1.51.Final</version>
	        </dependency>
	    </dependencies>
	
	</project>



## 测试

依次启动EchoServer，EchoClient


控制台分别打印：

### EchoServer

	Echo服务器启动
	服务端收到数据：wangzheyi is testing!
	读取完毕

### EchoClient

	Clent Actived
	Client recived :wangzheyi is testing!
	Client Completed