#单图片推理脚本：
#输入：一张原始图片 (6.jpg)
#输出：分割掩码 (output.jpg)，其中前景是灰色(150)，背景是黑色(0)

import torch
import numpy as np

from PIL import Image

img_size = (256, 256)
unet = torch.load('./ckpts/unet_epoch_99.pth', weights_only=False)  # 使用训练100个epoch后的模型

unet.eval() #将模型设置为评估模式（Evaluation Mode）


im = np.asarray(Image.open('data/JPEGImages/6.jpg').resize(img_size))  #加载图像 → 形状为 (256, 256, 3)，格式：(高度, 宽度, 通道数)

# 因为(PyTorch 的神经网络模型期望输入的张量格式为batch_size, channels, height, width)
im = im / 255.
im = im.transpose(2, 0, 1) #转置维度 → 形状变为 (3, 256, 256)，格式：(通道数, 高度, 宽度)
im = im[np.newaxis, :, :] #添加批次维度 → 形状变为 (1, 3, 256, 256)
im = im.astype('float32') #将NumPy数组的数据类型转换为32位浮点数（float32）
output = unet(torch.from_numpy(im)).detach().numpy()   #做了一系列数据转换NumPy数组(float32) → PyTorch张量 → 模型推理 → 分离梯度 → NumPy数组

output = np.squeeze(output) #移除数组中所有大小为1的维度。之前：形状可能是 (1, 1, 256, 256) 或 (1, 256, 256)，之后：形状变为 (256, 256)，样就得到了一个二维的分割掩码
output = np.where(output>0.5, 150, 0).astype(np.uint8)  #如果像素值 > 0.5，则设为150；否则设为0。astype(np.uint8)，转换为8位无符号整数类型
print(output.shape, type(output))
im = Image.fromarray(output)
im.save('output.jpg')
