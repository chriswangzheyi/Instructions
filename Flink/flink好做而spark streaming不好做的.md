# flink好做而spark streaming不好做的

1.全局去重，全局聚合操作，比如distinct ，uv等业务场景。flink适合，spark streaming做起来比较麻烦，后者要借助状态算子或者第三方存储，比如redis，alluxio等。

2.开窗操作且要求同一个窗口多次输出。这个可以用flink的trigger，spark streaming比较麻烦。

3.仅一次处理。spark streaming实现仅一次处理大部分都是依赖于输出端的幂等性。而flink，可以通过其分布式checkpoint的性质结合sink的事物来实现，也即分布式两段提交协议。当然，flink也可以利用sink的幂等性来实现仅一次处理。

4.更容易实现ddl，dml等完整的sql支持，进而实现完全sql实现业务开发，类似blink。spark streaming需要微批rdd转化为表，也是一个临时小表，不是全局的。

5.状态管理。flink可以方便地使用文件后端实现大状态管理，但是频繁发作也会引发linux系统操作文件的一些bug。当然，spark streaming可以灵活的使用第三方接口比如alluxio等也很方便。