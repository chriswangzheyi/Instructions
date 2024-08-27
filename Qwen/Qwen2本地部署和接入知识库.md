# Qwen2本地部署和接入知识库



## 下载LM studio



https://lmstudio.ai/



##  下载Qwen2 聊天模型



https://www.modelscope.cn/models/qwen/qwen2-7b-instruct-gguf/files

![](Images/1.png)



创建一个路径 

```
/modles/Qwen/Qwen2-7B-Instruct-GGUF
```

把qwen2-7b-instruct-q5_0.gguf 文件放进去



## LM studio中加载模型



打开LM studio，点击左侧“my models”, 选择模型本地路径“ models folder/Users/zheyiwang/Documents/models”，选择模型，保存。

![](Images/2.png)





## LM studio中选择chat配置



点击chat，右侧将Flash attention 勾选上。避免乱码

![](Images/3.png)



再在界面上方加载模型Qwen2

![](Images/4.png)

重新加载模型，配置就生效。



## 配置Ollama

为了分析文档，需要下载Ollma并安装

```
https://ollama.com/
```



命令行启动ollama:

```
ollama run llama3
```



命令行下载分词器：

```
ollama pull aerok/acge_text_embedding
```





## 启动 LM studio server



回到LM studio，启动server

![](Images/7.png)





获取服务器地址：

![](Images/8.png)







## 配置anythingLLM



下载anythingllm并安装

```
https://anythingllm.com/download
```





打开后，点击左下角设置

![](Images/5.png)







### 配置LLM提供者和URL



在anythingLLM中，选择LLM provider是LM studio， 填入获取的URL

![](Images/6.png)



点击保存配置。



## 配置Embedding

在anythingLLM中，配置embedding 为acge_text_embedding

![](Images/9.png)





## 本地知识库搭建



在anythingLLM中新建workspace，点击上传

![](Images/10.png)







上传文件或者URL



![](Images/11.png)





选择完毕后，加入右侧



![](Images/12.png)



点击保存。





## 知识库问答



上传的文档内有这些内容：

![](Images/13.png)



跟大模型提问,就可以获取内容：



![](Images/14.png)

