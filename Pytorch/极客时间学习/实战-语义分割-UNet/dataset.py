# 完成的数据流程：
# 读取时候：1.jpg (800×600) → resize → (32×32×3) → 归一化/255 → (32×32×3)
# 返回时候：(32×32×3) → transpose → (3×32×32)  ← 给模型训练用


import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image 


class CatSegmentationDataset(Dataset):
    
    # 模型输入是3通道数据（RGB彩色图像）
    in_channels = 3
    # 模型输出是1通道数据（二值分割掩码）
    out_channels = 1

    def __init__(
        self,
        images_dir,
        image_size=32,
    ):

        print("Reading images...")
        # 原图所在的位置（JPEGImages文件夹存放原始图像）
        image_root_path = images_dir + os.sep + 'JPEGImages'
        # Mask所在的位置（SegmentationClassPNG文件夹存放分割标注）
        mask_root_path = images_dir + os.sep + 'SegmentationClassPNG'
        # 将图片与Mask读入后，分别存在image_slices与mask_slices中
        self.image_slices = []
        self.mask_slices = []
        
        # 遍历图像文件夹中的所有图像
        for im_name in os.listdir(image_root_path):
            
            # 根据图像文件名构造对应的mask文件名（将扩展名改为.png）
            mask_name = im_name.split('.')[0] + '.png' 

            # 构造图像和mask的完整路径
            image_path = image_root_path + os.sep + im_name
            mask_path = mask_root_path + os.sep + mask_name

            # 读取图像并调整到指定大小，然后转换为numpy数组
            im = np.asarray(Image.open(image_path).resize((image_size, image_size)))
            # 读取mask并调整到指定大小，然后转换为numpy数组
            mask = np.asarray(Image.open(mask_path).resize((image_size, image_size)))
            
            # 将图像归一化到[0, 1]范围，便于神经网络训练
            self.image_slices.append(im / 255.)
            # 保存mask数据（通常mask值为0或255，表示背景和前景）
            self.mask_slices.append(mask)

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):

        # 根据索引获取对应的图像和掩码
        image = self.image_slices[idx] 
        mask = self.mask_slices[idx] 

        # 将图像从(H, W, C)转换为(C, H, W)格式，符合PyTorch的输入要求
        image = image.transpose(2, 0, 1)
        # 为mask添加通道维度，从(H, W)转换为(1, H, W)
        mask = mask[np.newaxis, :, :]

        # 确保数据类型为float32，符合PyTorch训练要求
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        return image, mask
