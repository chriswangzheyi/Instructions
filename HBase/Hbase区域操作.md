# Hbase区域操作

##查看区域

	scan 'hbase:meta'

![](../Images/5.png)

startkey 和 endkey表示了 region的分布情况。

## 切割区域

	split 'namespace','行号'

例如：

	split 'wangzheyi:test','row10'


分割后的日志：

	 wangzheyi:test,,1595686978043.9469a26df0b225 column=info:regioninfo, timestamp=1595686979458, value={ENCODED => 9469a26df0b225a7c89cd28437c711bb, NAME => 'wangzheyi:test,,159568
	 a7c89cd28437c711bb.                          6978043.9469a26df0b225a7c89cd28437c711bb.', STARTKEY => '', ENDKEY => 'row10'}   

	 wangzheyi:test,row10,1595686978043.b7c5fbf00 column=info:regioninfo, timestamp=1595686979487, value={ENCODED => b7c5fbf0091195d6e8bffdd6b29fc85e, NAME => 'wangzheyi:test,row10,1
	 91195d6e8bffdd6b29fc85e.                     595686978043.b7c5fbf0091195d6e8bffdd6b29fc85e.', STARTKEY => 'row10', ENDKEY => ''}     

可以看到startkey 和 endkey有所区别

**value值记录了存放的Datanode**



## 合并区域

语法：

	merge_region 'encoded_regionanme','encoded_regionanme'

![](../Images/6.png)

区域的ENCODED值可以通过日志查看到。