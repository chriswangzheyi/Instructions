```python
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
```


```python
train_data = datasets.MNIST(
    root="data/minst",
    train=True,
    transform= transforms.ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root="data/minst",
    train=False,
    transform= transforms.ToTensor(),
    download=True
)
```

# 数据加载器


```python
batch_size = 100
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False
)
```


```python
# 构建网络
```


```python
class Model(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__(),
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,x):
        logits = self.linear(x)
        return logits
```


```python
input_size = 28*28
output_size = 10
model = Model(input_size,output_size)
```

## 定义损失函数和优化器


```python
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
```

## 模型评估


```python
def evaluate(model, data_loader):
    model.eval() #eval() 方法用于将模型设置为评估模式（evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():# 在评估的时候，不需要梯度
        for x,y in data_loader:
            x = x.view(-1,input_size)
            logits = model(x)
            _,predicted = torch.max(logits.data,1)
            total += y.size(0)
            correct += (predicted ==y).sum().item()
    return correct / total
```


```python
#训练模型
```


```python
for epoch in range(20):
    for images, labels in train_loader:
        #将图像和标签转换为张量
        images = images.view(-1,28*28)

        #前向传播
        outputs = model(images)
        loss = criterion(outputs,labels)

        #反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accurancy = evaluate(model,test_loader)
    print(f'Epoch {epoch+1}: test accurancy = {accurancy:2f}')
```

    Epoch 1: test accurancy = 0.905700
    Epoch 2: test accurancy = 0.907100
    Epoch 3: test accurancy = 0.907800
    Epoch 4: test accurancy = 0.908700
    Epoch 5: test accurancy = 0.910500
    Epoch 6: test accurancy = 0.910000
    Epoch 7: test accurancy = 0.910200
    Epoch 8: test accurancy = 0.911500
    Epoch 9: test accurancy = 0.912200
    Epoch 10: test accurancy = 0.912900
    Epoch 11: test accurancy = 0.912900
    Epoch 12: test accurancy = 0.913500
    Epoch 13: test accurancy = 0.914600
    Epoch 14: test accurancy = 0.915000
    Epoch 15: test accurancy = 0.914600
    Epoch 16: test accurancy = 0.915300
    Epoch 17: test accurancy = 0.915400
    Epoch 18: test accurancy = 0.915600
    Epoch 19: test accurancy = 0.916700
    Epoch 20: test accurancy = 0.916300

