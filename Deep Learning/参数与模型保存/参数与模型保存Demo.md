```python
import torch
```

# 单个张量的保存与读取


```python
a = torch.rand(6)
a
```




    tensor([0.1503, 0.2339, 0.1137, 0.5502, 0.7201, 0.4896])




```python
torch.save(a,"model/tensor_a")
```


```python
torch.load("model/tensor_a")
```




    tensor([0.1503, 0.2339, 0.1137, 0.5502, 0.7201, 0.4896])



# 多个张量的保存与读取


```python
a = torch.rand(6)
b = torch.rand(6)
c = torch.rand(6)
```


```python
torch.save([a,b,c],"model/tensor_abc")
```


```python
torch.load("model/tensor_abc")
```




    [tensor([0.0238, 0.4387, 0.9022, 0.9829, 0.7061, 0.1464]),
     tensor([0.1474, 0.2188, 0.5963, 0.7127, 0.7847, 0.4968]),
     tensor([0.7661, 0.1341, 0.1468, 0.4066, 0.7826, 0.7857])]



# 字典形式保存与读取


```python
tensor_dict={'a':a,'b':b,'c':c}
tensor_dict
```




    {'a': tensor([0.0238, 0.4387, 0.9022, 0.9829, 0.7061, 0.1464]),
     'b': tensor([0.1474, 0.2188, 0.5963, 0.7127, 0.7847, 0.4968]),
     'c': tensor([0.7661, 0.1341, 0.1468, 0.4066, 0.7826, 0.7857])}




```python
torch.save(tensor_dict,"model/tensor_dict")
```


```python
torch.load("model/tensor_dict")
```




    {'a': tensor([0.0238, 0.4387, 0.9022, 0.9829, 0.7061, 0.1464]),
     'b': tensor([0.1474, 0.2188, 0.5963, 0.7127, 0.7847, 0.4968]),
     'c': tensor([0.7661, 0.1341, 0.1468, 0.4066, 0.7826, 0.7857])}



# 模型的保存与加载


```python
import torch
import torch.nn as nn

# 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(5, 2)   # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 隐藏层的激活函数ReLU
        x = self.fc2(x)              # 输出层
        return x

# 创建模型实例
model = SimpleNet()

# 保存模型
torch.save(model.state_dict(), 'model/model.pth')

# 加载模型
model_loaded = SimpleNet()
model_loaded.load_state_dict(torch.load('model/model.pth'))

# 使用加载的模型进行预测
input_tensor = torch.randn(1, 10)  # 构造一个输入张量
output = model_loaded(input_tensor)  # 前向传播
print("预测结果:", output)
```

    预测结果: tensor([[-0.2824, -0.0339]], grad_fn=<AddmmBackward0>)



```python
# Checkpoint
```


```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(5, 2)   # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 隐藏层的激活函数ReLU
        x = self.fc2(x)              # 输出层
        return x

# 创建模型实例
model = SimpleNet()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 模拟训练过程
for epoch in range(10):
    # 假设每个 epoch 包含 1000 个批次
    for batch in range(1000):
        # 模拟一个批次的输入数据和目标标签
        input_tensor = torch.randn(64, 10)
        target_tensor = torch.randint(0, 2, (64,))
        
        # 前向传播
        output = model(input_tensor)
        loss = nn.CrossEntropyLoss()(output, target_tensor)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每 100 个批次保存一次模型和优化器状态
        if batch % 100 == 0:
            # 保存模型状态字典和优化器状态字典到文件
            torch.save({
                'epoch': epoch,
                'batch': batch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, 'checkpoint.pth')

```


```python

```
