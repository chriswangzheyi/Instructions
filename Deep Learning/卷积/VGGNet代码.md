# 导包


```python
import torch
import torch.nn as nn
from torchinfo import summary
```

# 网络模型定义


```python
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        # 卷积层直接使用传入的结构，后面有专门构建这部分的内容
        self.features = features
        # 定义全连接层
        self.classifier = nn.Sequential(
            # 全连接层+ReLU+Dropout
            nn.Linear(512 * 7 * 7, 4096), #512 个通道，每个通道大小为 7x7。一般采用4096维的输出作为中间层的维度
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # 全连接层 + ReLU + Dropout
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),  # inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            nn.Dropout(),
            # 全连接层
            nn.Linear(4096, num_classes),
        )

    # 定义前向传播函数
    def forward(self, x):
        # 先经过feature提取特征，flatten后送入全连接层
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

# 定义配置项


```python
# 定义相关配置项，其中M表示池化层，数值完全和论文中保持一致
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

```

每个列表都是由数字和字符构成的序列，数字代表卷积层中的输出通道数，字符 'M' 代表最大池化层。例如，'vgg11' 的列表 [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'] 描述了一个具有11层的VGG网络结构。

下面是这些VGG网络的简单描述：

- 'vgg11': 这个版本有8个卷积层和3个全连接层。
- 'vgg13': 这个版本有10个卷积层和3个全连接层。
- 'vgg16': 这个版本有13个卷积层和3个全连接层。
- 'vgg19': 这个版本有16个卷积层和3个全连接层。



# 拼接卷积层


```python
# 根据传入的配置项拼接卷积层
def make_layers(cfg):
    layers = []
    in_channels = 3  # 初始通道数为3
    # 遍历传入的配置项
    for v in cfg:
        if v == 'M':  # 如果是池化层，则直接新增MaxPool2d即可
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:  # 如果是卷积层，则新增3x3卷积 + ReLU非线性激活
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # 记录通道数，作为下一次的in_channels *******
    # 返回使用Sequential构造的卷积层
    return nn.Sequential(*layers)
```



# 封装函数


```python
# 封装函数，依次传入对应的配置项
def vgg11(num_classes=1000):
    return VGG(make_layers(cfgs['vgg11']), num_classes=num_classes)


def vgg13(num_classes=1000):
    return VGG(make_layers(cfgs['vgg13']), num_classes=num_classes)


def vgg16(num_classes=1000):
    return VGG(make_layers(cfgs['vgg16']), num_classes=num_classes)


def vgg19(num_classes=1000):
    return VGG(make_layers(cfgs['vgg19']), num_classes=num_classes)


```

# 查看网络结构


```python
# 网络结构
# 查看模型结构即参数数量，input_size  表示示例输入数据的维度信息
summary(vgg16(), input_size=(1, 3, 224, 224))

```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    VGG                                      [1, 1000]                 --
    ├─Sequential: 1-1                        [1, 512, 7, 7]            --
    │    └─Conv2d: 2-1                       [1, 64, 224, 224]         1,792
    │    └─ReLU: 2-2                         [1, 64, 224, 224]         --
    │    └─Conv2d: 2-3                       [1, 64, 224, 224]         36,928
    │    └─ReLU: 2-4                         [1, 64, 224, 224]         --
    │    └─MaxPool2d: 2-5                    [1, 64, 112, 112]         --
    │    └─Conv2d: 2-6                       [1, 128, 112, 112]        73,856
    │    └─ReLU: 2-7                         [1, 128, 112, 112]        --
    │    └─Conv2d: 2-8                       [1, 128, 112, 112]        147,584
    │    └─ReLU: 2-9                         [1, 128, 112, 112]        --
    │    └─MaxPool2d: 2-10                   [1, 128, 56, 56]          --
    │    └─Conv2d: 2-11                      [1, 256, 56, 56]          295,168
    │    └─ReLU: 2-12                        [1, 256, 56, 56]          --
    │    └─Conv2d: 2-13                      [1, 256, 56, 56]          590,080
    │    └─ReLU: 2-14                        [1, 256, 56, 56]          --
    │    └─Conv2d: 2-15                      [1, 256, 56, 56]          590,080
    │    └─ReLU: 2-16                        [1, 256, 56, 56]          --
    │    └─MaxPool2d: 2-17                   [1, 256, 28, 28]          --
    │    └─Conv2d: 2-18                      [1, 512, 28, 28]          1,180,160
    │    └─ReLU: 2-19                        [1, 512, 28, 28]          --
    │    └─Conv2d: 2-20                      [1, 512, 28, 28]          2,359,808
    │    └─ReLU: 2-21                        [1, 512, 28, 28]          --
    │    └─Conv2d: 2-22                      [1, 512, 28, 28]          2,359,808
    │    └─ReLU: 2-23                        [1, 512, 28, 28]          --
    │    └─MaxPool2d: 2-24                   [1, 512, 14, 14]          --
    │    └─Conv2d: 2-25                      [1, 512, 14, 14]          2,359,808
    │    └─ReLU: 2-26                        [1, 512, 14, 14]          --
    │    └─Conv2d: 2-27                      [1, 512, 14, 14]          2,359,808
    │    └─ReLU: 2-28                        [1, 512, 14, 14]          --
    │    └─Conv2d: 2-29                      [1, 512, 14, 14]          2,359,808
    │    └─ReLU: 2-30                        [1, 512, 14, 14]          --
    │    └─MaxPool2d: 2-31                   [1, 512, 7, 7]            --
    ├─Sequential: 1-2                        [1, 1000]                 --
    │    └─Linear: 2-32                      [1, 4096]                 102,764,544
    │    └─ReLU: 2-33                        [1, 4096]                 --
    │    └─Dropout: 2-34                     [1, 4096]                 --
    │    └─Linear: 2-35                      [1, 4096]                 16,781,312
    │    └─ReLU: 2-36                        [1, 4096]                 --
    │    └─Dropout: 2-37                     [1, 4096]                 --
    │    └─Linear: 2-38                      [1, 1000]                 4,097,000
    ==========================================================================================
    Total params: 138,357,544
    Trainable params: 138,357,544
    Non-trainable params: 0
    Total mult-adds (G): 15.48
    ==========================================================================================
    Input size (MB): 0.60
    Forward/backward pass size (MB): 108.45
    Params size (MB): 553.43
    Estimated Total Size (MB): 662.49
    ==========================================================================================



# 使用torchvision定义模型


```python
# 简单实现 ---> 利用torch.vision
# 查看torchvision自带的模型结构即参数量
from torchvision import models
summary(models.vgg16(),input_size=(1,3,224,224))

```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    VGG                                      [1, 1000]                 --
    ├─Sequential: 1-1                        [1, 512, 7, 7]            --
    │    └─Conv2d: 2-1                       [1, 64, 224, 224]         1,792
    │    └─ReLU: 2-2                         [1, 64, 224, 224]         --
    │    └─Conv2d: 2-3                       [1, 64, 224, 224]         36,928
    │    └─ReLU: 2-4                         [1, 64, 224, 224]         --
    │    └─MaxPool2d: 2-5                    [1, 64, 112, 112]         --
    │    └─Conv2d: 2-6                       [1, 128, 112, 112]        73,856
    │    └─ReLU: 2-7                         [1, 128, 112, 112]        --
    │    └─Conv2d: 2-8                       [1, 128, 112, 112]        147,584
    │    └─ReLU: 2-9                         [1, 128, 112, 112]        --
    │    └─MaxPool2d: 2-10                   [1, 128, 56, 56]          --
    │    └─Conv2d: 2-11                      [1, 256, 56, 56]          295,168
    │    └─ReLU: 2-12                        [1, 256, 56, 56]          --
    │    └─Conv2d: 2-13                      [1, 256, 56, 56]          590,080
    │    └─ReLU: 2-14                        [1, 256, 56, 56]          --
    │    └─Conv2d: 2-15                      [1, 256, 56, 56]          590,080
    │    └─ReLU: 2-16                        [1, 256, 56, 56]          --
    │    └─MaxPool2d: 2-17                   [1, 256, 28, 28]          --
    │    └─Conv2d: 2-18                      [1, 512, 28, 28]          1,180,160
    │    └─ReLU: 2-19                        [1, 512, 28, 28]          --
    │    └─Conv2d: 2-20                      [1, 512, 28, 28]          2,359,808
    │    └─ReLU: 2-21                        [1, 512, 28, 28]          --
    │    └─Conv2d: 2-22                      [1, 512, 28, 28]          2,359,808
    │    └─ReLU: 2-23                        [1, 512, 28, 28]          --
    │    └─MaxPool2d: 2-24                   [1, 512, 14, 14]          --
    │    └─Conv2d: 2-25                      [1, 512, 14, 14]          2,359,808
    │    └─ReLU: 2-26                        [1, 512, 14, 14]          --
    │    └─Conv2d: 2-27                      [1, 512, 14, 14]          2,359,808
    │    └─ReLU: 2-28                        [1, 512, 14, 14]          --
    │    └─Conv2d: 2-29                      [1, 512, 14, 14]          2,359,808
    │    └─ReLU: 2-30                        [1, 512, 14, 14]          --
    │    └─MaxPool2d: 2-31                   [1, 512, 7, 7]            --
    ├─AdaptiveAvgPool2d: 1-2                 [1, 512, 7, 7]            --
    ├─Sequential: 1-3                        [1, 1000]                 --
    │    └─Linear: 2-32                      [1, 4096]                 102,764,544
    │    └─ReLU: 2-33                        [1, 4096]                 --
    │    └─Dropout: 2-34                     [1, 4096]                 --
    │    └─Linear: 2-35                      [1, 4096]                 16,781,312
    │    └─ReLU: 2-36                        [1, 4096]                 --
    │    └─Dropout: 2-37                     [1, 4096]                 --
    │    └─Linear: 2-38                      [1, 1000]                 4,097,000
    ==========================================================================================
    Total params: 138,357,544
    Trainable params: 138,357,544
    Non-trainable params: 0
    Total mult-adds (G): 15.48
    ==========================================================================================
    Input size (MB): 0.60
    Forward/backward pass size (MB): 108.45
    Params size (MB): 553.43
    Estimated Total Size (MB): 662.49
    ==========================================================================================



# 模型训练与评估


```python
import torch.optim as optim
from torch.utils.data  import DataLoader
from torchvision import datasets,transforms,models
from tqdm import *
import numpy as np
import sys

```

# 设备检测


```python
device = torch.device("mps")
```



# 设置随机数种子


```python
# 设置随机数种子
torch.manual_seed(0)
```




    <torch._C.Generator at 0x116374590>



# 定义模型、优化器、损失函数


```python
# 定义模型、优化器、损失函数
model = vgg11(num_classes=102).to(device)
optimizer = optim.SGD(model.parameters(),lr=0.002,momentum=0.9)
criterion = nn.CrossEntropyLoss()
```

# 设置训练集的数据转换


```python
trainform_train = transforms.Compose([
    transforms.RandomRotation(30),# 随机旋转-30到30度之间
    transforms.RandomResizedCrop((224,224)), # 随机比例裁剪并进行resize
    transforms.RandomHorizontalFlip(p=0.5), #随机垂直翻转
    transforms.ToTensor(), # 将数据转换成张量
    # 对三通道数据进行归一化（均值，标准差），数值是冲ImageNet数据集上的百万张图片中随机抽样计算得到
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
```

# 设置测试集的数据变换


```python
#设置测试集的数据变换，不进行数据增强，仅仅使用resize和归一化操作
transform_test = transforms.Compose([
    transforms.Resize((224,224)), #resize
    transforms.ToTensor(),#将数据转换成张量形式
    # 对三通道数据进行归一化（均值，标准差），数值是冲ImageNet数据集上的百万张图片中随机抽样计算得到
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

```

# 加载训练数据


```python
# 加载训练数据，需要特别注意的是Flowers102的数据集，test簇的数据集较多一些，所以这里使用test作为训练集
train_dataset = datasets.Flowers102(root='data/flowers102',split="test",download=True,transform=trainform_train)
```

# 实例化训练数据加载器


```python
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=4)
```

# 加载测试数据


```python
test_dataset = datasets.Flowers102(root='data/flowers102',split="train",download=True,transform=transform_test)
```

# 实例化测试数据加载器


```python
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=4)
```

# 设置epochs并进行训练


```python
# 设置epochs并进行训练
num_epochs = 200 # 设置epoch数
loss_history = [] # 创建损失历史记录表
acc_history = [] # 创建准确率历史记录表
# tqdm用于显示进度条并评估任务时间开销
for epoch in tqdm(range(num_epochs),file=sys.stdout):
    # 记录损失和预测正确数
    total_loss = 0
    total_correct = 0

    # 批量训练
    model.train()
    for inputs,labels in train_loader:
        # 将数据转移到指定计算资源设备上
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 预测、损失函数、反向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        #记录训练集loss
        total_loss += loss.item()
    # 测试模型，不计算梯度
    model.eval()
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 预测
            outputs = model(inputs)
            # 记录测试集预测正确数
            total_correct  += (outputs.argmax(1) == labels).sum().item()
    # 记录训练集损失和测试集准确率
    loss_history.append(np.log10(total_loss)) # 将损失加入到损失历史记录列表中，由于数值有时比较大所以这里取了对数
    acc_history.append(total_correct/len(test_dataset)) # 将准确率加入到准确率历史记录列表

    # 打印中间值
    if epoch % 10 == 0:
        tqdm.write("Epoch:{0} Loss:{1} ACC:{2}".format(epoch,loss_history[-1],acc_history[-1]))

```

    Epoch:0 Loss:2.651430075296246 ACC:0.00980392156862745
    Epoch:10 Loss:2.62598004902669 ACC:0.01764705882352941
      7%|▋         | 14/200 [28:44<6:20:18, 122.68s/it]

# 使用Matplotlib绘制损失和准确率的曲线图


```python
import matplotlib.pyplot as plt
plt.plot(loss_history,label='loss')
plt.plot(acc_history,label='accuracy')
plt.legend()
plt.show()
```

# 输出准确率


```python
print("Accuracy:",acc_history[-1])
```
