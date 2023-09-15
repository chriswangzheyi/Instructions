# Transformer 简介
参考：https://www.bilibili.com/video/BV1ih4y1J7rx/?spm_id_from=333.337.search-card.all.click&vd_source=a267912a9cbb36e71ab719af20b7a8cb


## RNN & GRU/LSTM 和 Attention Mechanism 对比

###RNN & GRU/LSM

短时记忆窗口

![](Images/1.png)

![](Images/2.png)

### Attention Mechanism

无限长度的窗口

![](Images/3.png)


## Transformer 架构

![](Images/4.png)

### 解释

编码器将输入序列映射到一个抽象的连续表示，其中保护了该输入的所有学习信息。解码器接着获取这个连续表示，并逐步生成单一输出，同时还会输入前一个输出。

![](Images/5.png)

### 例子

#### 步骤1： Input Embedding

Embedding 可以看作是一个查找表，用来获取每个单词的学习向量表示。神经网络通过数字进行学习，以每个单子都映射到一个连续值的向量。

![](Images/6.png)

#### 步骤2：Positional Encoding

因为Transformer编码器没有像RNN的递归，需要将位置信息添加到输入嵌入中，同归位置编码实现的

![](Images/7.png)

对于每个奇数时间步，使用余弦函数创建一个向量。对于每个偶数时间步，使用正弦函数创造一个向量。然后将这些向量添加到相应的Eembedding向量中



#### Encoder layer


![](Images/8.png)

#### 步骤3:  Multi-headed Atention

##### 步骤3.1 Self-Attention

![](Images/9.png)

Self-Attention 允许模型将输入的每个单词和其他单词关联起来。

![](Images/10.png)

将单词分别送入三个不同的全链接层，创建query， key， value向量。 query矩阵和key矩阵经过点积乘法产生了一个scores矩阵

![](Images/11.png)

scores 矩阵确定了一个单词应该如何关注其他单词。分数越高，关注度越高。

![](Images/12.png)

得分缩放：

![](Images/13.png)

![](Images/14.png)

输入向量

![](Images/15.png)

为了使计算成为Multi-headed 计算，需要在应用自注意力之前将query，key，value分为n个向量。每个自注意力称为一个head，每个head会禅师一个输出向量，这些向量在经过最后的Linear层之前被拼接成一个向量。

![](Images/16.png)

![](Images/17.png)

multi-head的输出向量加到原始输入上，这叫残差连接。残差连接经过LayerNorm归一化

![](Images/18.png)

残差连接有助于网络训练，因为他允许梯度直接流过网络，显著减少所需的训练时间。

![](Images/19.png)

以上操作都是为了将输入编码为连续表示，带有注意力信息。这将帮助decoer在解码过程中关注输入中的适当词汇。可以将encoder堆叠nci以进一步编码信息，其中每一层都有机会学习不同的注意力表示，从而有可能提高transformer网络的预测能力。


### 解码器

解码器的任务是生成文本序列。解码器是自回归的，

![](Images/21.png)

将先前输出的列表作为输入，以及包含来自输入的注意力信息的编码输出。当解码器生成一个结束标记作为输出时，解码停止。


#### 步骤5:  Output Embedding & Positional Encoding

![](Images/20.png)


#### 步骤6:  Decoder Multi-Headed Attention 1

解码器的multi-headed attention 的工作方式稍有不同。因为解码器是自回归的，并且逐词生成序列。例如，在计算单词“am”的注意力得分时，不应该访问单词“find”，因为那个词时在之后生成的未来词。


![](Images/22.png)

需要一种方法来计算未来单词的注意力得分，这种方法称之为Mask。


![](Images/23.png)


![](Images/24.png)

以I为例，只跟自己相关，对后面的单词都不相关。

在这一层仍然是multi-headed，输出的结果是一个带有掩码的输出向量。允许解码器决定哪个编码器输入是相关的焦点。

![](Images/25.png)


#### 第二个multi-headed

![](Images/26.png)

编码器的输出时query 和 key


#### 步骤7： Linear CLassifier

![](Images/27.png)

该层充当分类器。分类器的小心跟类别数量相同。例如，有10000个类别，表示10000个单词，那么分类器的输出大小是10000.分类器的输出然后被送入一个softmax层。softmax为每个类别生成0-1之间的概率得分。取概率索引得分最高着为预测单词。






















