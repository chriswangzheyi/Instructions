# InputStream和Reader的区别

java.io下面有两个抽象类：InputStream和Reader

* InputStream：得到的是字节输入流
* Reader：读取的是字符流

**InputStream提供的是字节流的读取，而非文本读取，这是和Reader类的根本区别。**

即用Reader读取出来的是char数组或者String ，使用InputStream读取出来的是byte数组。