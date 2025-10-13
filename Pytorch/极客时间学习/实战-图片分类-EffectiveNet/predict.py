# 预测脚本：使用训练好的模型对新图片进行分类预测

import torchvision
from efficientnet import EfficientNet
import torch
import torch.nn as nn
from PIL import Image
import argparse
from dataset import build_transform

if __name__ == '__main__':
    # 设置命令行参数：接收要预测的图片路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', default='./data/train/logo/06.jpeg', help='图片路径（默认：./data/train/logo/06.jpeg）')
    args = parser.parse_args()
    
    # 设置设备：优先使用 MPS（Mac GPU）> CUDA（NVIDIA GPU）> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"=> 使用设备: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"=> 使用设备: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print(f"=> 使用设备: CPU")
    
    # 创建模型结构（从零开始，还没有训练好的权重）
    model = EfficientNet.from_name('efficientnet-b0')
    # 获取最后一层的输入特征数量
    num_ftrs = model._fc.in_features
    # 替换最后一层：改为输出2个类别（logo 和 others）
    model._fc = nn.Linear(num_ftrs, 2)
    
    # 加载训练好的模型权重（第9轮保存的模型），并映射到当前设备
    model.load_state_dict(torch.load('./ckpts/checkpoint.pth.tar.epoch_9', map_location=device))
    # 将模型移到对应设备
    model = model.to(device)
    # 切换到评估模式（关闭训练特性）
    model.eval()
    
    # 读取要预测的图片
    image = Image.open(args.path)
    # 转换为RGB格式（防止是灰度图）
    image = image.convert('RGB')
    
    # 图片预处理：调整大小、转张量、归一化
    transform = build_transform(224)
    # 应用预处理并增加批次维度 [3,224,224] -> [1,3,224,224]
    input_tensor = transform(image).unsqueeze(0)
    # 将数据移到对应设备
    input_tensor = input_tensor.to(device)
    
    # 模型预测：得到每个类别的得分
    with torch.no_grad():  # 预测时不需要计算梯度，节省内存
        pred = model(input_tensor)
    
    # 获取预测结果
    class_names = ['logo', 'others']
    pred_scores = pred[0].cpu()  # 移到 CPU 用于打印
    pred_class = pred.argmax().item()  # 得分最高的类别索引
    
    # 输出友好的结果
    print(f"\n图片: {args.path}")
    print(f"预测结果: {class_names[pred_class]}")
    print(f"置信度: logo={pred_scores[0]:.4f}, others={pred_scores[1]:.4f}")
