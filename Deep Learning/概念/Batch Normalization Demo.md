```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定义是否使用GPU
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 定义没有批归一化的模型
class NetNoBatchNorm(nn.Module):
    def __init__(self):
        super(NetNoBatchNorm, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义带有批归一化的模型
class NetWithBatchNorm(nn.Module):
    def __init__(self):
        super(NetWithBatchNorm, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
```



```python
# 训练函数
def train(net, criterion, optimizer, num_epochs=5):
    net.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")
```


```python
# 测试函数
def test(net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

```


```python
# 训练没有批归一化的模型
print("Training model without batch normalization:")
net_no_batch_norm = NetNoBatchNorm().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net_no_batch_norm.parameters(), lr=0.001)
train(net_no_batch_norm, criterion, optimizer)
test(net_no_batch_norm)
```

输出：

    Training model without batch normalization:
    Epoch 1, Loss: 0.3643367748731362
    Epoch 2, Loss: 0.17297070974638976
    Epoch 3, Loss: 0.12652325319714033
    Epoch 4, Loss: 0.1029968868369169
    Epoch 5, Loss: 0.08415257295261996
    Accuracy: 96.84%



```python
# 训练带有批归一化的模型
print("\nTraining model with batch normalization:")
net_with_batch_norm = NetWithBatchNorm().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net_with_batch_norm.parameters(), lr=0.001)
train(net_with_batch_norm, criterion, optimizer)
test(net_with_batch_norm)
```

输出：


    Training model with batch normalization:
    Epoch 1, Loss: 0.22415152790823153
    Epoch 2, Loss: 0.09037712990328162
    Epoch 3, Loss: 0.0660873775018939
    Epoch 4, Loss: 0.05121190874164165
    Epoch 5, Loss: 0.04223891761119682
    Accuracy: 98.01%


