# geekTime-image-classification

使用 [EfficientNet](https://github.com/qubvel/efficientnet) 构建一个图像分类模型，用于识别 logo 和 others 两个类别。

## 项目结构

```
geekTime-image-classification-main/
├── data/                      # 数据集目录
│   ├── train/                 # 训练集
│   │   ├── logo/             # logo 类别图片
│   │   └── others/           # others 类别图片
│   └── val/                   # 验证集
│       ├── logo/             # logo 类别图片
│       └── others/           # others 类别图片
├── efficientnet/              # EfficientNet 模型实现
│   ├── __init__.py           
│   ├── model.py              # 模型定义
│   └── utils.py              # 工具函数
├── ckpts/                     # 模型权重保存目录（训练后生成）
├── dataset.py                 # 数据集处理模块
├── train.py                   # 训练脚本
├── predict.py                 # 预测脚本
└── README.md                  # 项目说明文档
```

## 文件说明

### 核心文件

- **`dataset.py`** - 数据集处理模块
  - 构建图像预处理流程（调整大小、转张量、归一化）
  - 加载训练数据集

- **`train.py`** - 模型训练脚本
  - 创建 EfficientNet 模型（支持预训练）
  - 执行训练循环
  - 定期保存模型权重
  - 支持 GPU 加速（MPS/CUDA/CPU）

- **`predict.py`** - 模型预测脚本
  - 加载训练好的模型
  - 对新图片进行分类预测
  - 输出预测结果和置信度

### 模型文件

- **`efficientnet/`** - EfficientNet 模型实现
  - 基于 EfficientNet-B0 到 B7 架构
  - 支持预训练权重加载

## 环境安装

### 1. 创建虚拟环境（推荐）

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install torch torchvision
```

### 3. 验证环境

```bash
# Mac 用户检查 MPS 是否可用
python -c "import torch; print('MPS 可用:', torch.backends.mps.is_available())"

# 其他用户检查 CUDA 是否可用
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"
```

## 使用步骤

### 第一步：训练模型

运行训练脚本，模型会在训练数据上学习：

```bash
# 激活虚拟环境
source venv/bin/activate

# 开始训练
python train.py
```

**训练参数说明：**
- `--train-data`: 训练数据路径（默认：`./data/train`）
- `--image-size`: 图像尺寸（默认：224）
- `--batch-size`: 批次大小（默认：10）
- `--epochs`: 训练轮数（默认：10）
- `--lr`: 学习率（默认：0.001）
- `--checkpoint-dir`: 模型保存路径（默认：`./ckpts/`）
- `--save-interval`: 保存间隔（默认：每轮保存）
- `--arch`: 模型架构（默认：efficientnet-b0）
- `--pretrained`: 是否使用预训练模型（默认：True）

**示例：自定义参数训练**

```bash
python train.py --epochs 20 --batch-size 16 --lr 0.0001
```

**训练输出：**
```
=> 使用设备: MPS (Apple Silicon GPU)
=> using pre-trained model 'efficientnet-b0'
Epoch 0 tensor(0.6931)
Epoch 1 tensor(0.5234)
...
```

训练完成后，模型权重会保存在 `./ckpts/` 目录下，文件名格式为 `checkpoint.pth.tar.epoch_X`。

### 第二步：预测图片

使用训练好的模型对新图片进行预测：

```bash
# 激活虚拟环境
source venv/bin/activate

# 使用默认图片预测
python predict.py

# 预测指定图片
python predict.py --path ./your_image.jpg
```

**预测输出：**
```
=> 使用设备: MPS (Apple Silicon GPU)

图片: ./data/train/logo/06.jpeg
预测结果: logo
置信度: logo=0.3553, others=-0.2102
```

## 数据集格式

训练数据需要按照以下结构组织：

```
data/
├── train/              # 训练集
│   ├── logo/          # 第一个类别
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── others/        # 第二个类别
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── val/               # 验证集（可选）
    ├── logo/
    └── others/
```

- 每个子文件夹代表一个类别
- 文件夹名称会自动作为类别标签
- 支持常见图片格式（jpg、jpeg、png 等）

## 性能优化

### GPU 加速

代码已内置 GPU 加速支持：
- **Mac M1/M2/M3**: 自动使用 MPS（Metal Performance Shaders）
- **NVIDIA GPU**: 自动使用 CUDA
- **CPU**: 无 GPU 时自动使用 CPU

GPU 训练速度比 CPU 快 **5-100倍**！

### 训练速度建议

1. **增大批次大小** (`--batch-size`)：如果显存足够，可以设置为 16、32
2. **使用预训练模型** (`--pretrained True`)：收敛更快，效果更好
3. **减少训练轮数** (`--epochs`)：过多轮次可能导致过拟合

## 常见问题

### 1. 训练时内存不足
- 减小 `--batch-size` 参数（如改为 4 或 8）
- 减小 `--image-size` 参数（如改为 128）

### 2. 模型加载失败
- 确保 `./ckpts/checkpoint.pth.tar.epoch_9` 文件存在
- 检查是否完成了至少 9 轮训练

### 3. 预测结果不准确
- 增加训练轮数 (`--epochs`)
- 增加训练数据量
- 使用数据增强（需要修改 `dataset.py`）

## 技术栈

- **PyTorch**: 深度学习框架
- **TorchVision**: 图像处理库
- **EfficientNet**: 高效的卷积神经网络架构
- **PIL**: Python 图像处理库

## 参考资料

- [EfficientNet 论文](https://arxiv.org/abs/1905.11946)
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [图像分类入门教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
