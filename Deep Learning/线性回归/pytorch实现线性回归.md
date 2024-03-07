```python
import numpy as np
import torch
import torch.nn as nn

# 设置随机数种子，是的每次运行代码生成的数据相同
np.random.seed(42)

# 生成随机数
x= np.random.rand(100,1)
y= 1 + 2*x + 0.1* np.random.randn(100,1)

# 将数据转换为pytorch tensor
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()
```


```python
# 设置超惨
learning_rate = 0.1
num_epochs = 1000
```


```python
# 定义输入数据的维度和输出数据的维度
input_dim = 1
output_dim = 1
```


```python
# 定义模型
model = nn.Linear(input_dim,output_dim) #nn.Linear是全连接层（Fully Connected Layer）或密集层（Dense Layer）
```


```python
# 定义损失函数和优化器
criterion= nn.MSELoss() #nn.MSELoss() 是一个用于计算均方误差
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)  
#优化器的作用是通过调整模型的参数来最小化训练过程中定义的损失函数
# torch.optim.SGD 是 PyTorch 中实现随机梯度下降（Stochastic Gradient Descent，SGD）优化算法的一个类。
# torch.optim.SGD 类接受两个参数：params 和 lr。其中，params 是一个包含了模型参数的可迭代对象，通常通过调用 model.parameters() 方法获取，用于指定要更新的参数；lr 则是学习率。
```


```python
# 开始训练

for epoch in range(num_epochs):
    # 将输入数据喂给模型
    y_pred = model(x_tensor)
    
    #计算损失
    loss = criterion(y_pred, y_tensor)
    
    #清空梯度
    optimizer.zero_grad() #用于将优化器中所有参数的梯度清零，以确保每次参数更新时都使用当前轮迭代计算的梯度，而不是累积的梯度。
    
    #反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
# 输出训练后的参数
print('w:', model.weight.data)
print('b:', model.bias.data)
```

    w: tensor([[1.9540]])
    b: tensor([1.0215])
