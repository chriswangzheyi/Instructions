# 数据集处理模块

from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode


def _norm_advprop(img):
    """
    归一化函数：将图像像素值从 [0,1] 转换到 [-1,1]
    举例：
        0（最黑）→ 0×2-1 = -1
        0.5（灰色）→ 0.5×2-1 = 0
        1（最白）→ 1×2-1 = 1
    """
    return img * 2.0 - 1.0


def build_transform(dest_image_size):
    """
    构建图像预处理流程
    参数: dest_image_size - 目标图像尺寸（整数或元组）
    返回: 图像转换器，包含调整大小、转张量、归一化三个步骤
    """
    # 创建归一化转换器
    normalize = transforms.Lambda(_norm_advprop)    #transforms.Lambda可以把你的函数包装一下，让它能放进工具箱
    
    # 处理图像尺寸参数：支持整数（正方形）或元组（长方形）
    if not isinstance(dest_image_size, tuple):
        # 如果输入是整数（如 224），转换为正方形尺寸 (224, 224)
        dest_image_size = (dest_image_size, dest_image_size)
    else:
        # 如果输入已经是元组（如 (256, 128)），直接使用
        dest_image_size = dest_image_size

    # 组合多个图像转换步骤，图像会依次经过每一步处理
    transform = transforms.Compose([
        # 步骤1：调整图像到目标尺寸，使用双三次插值保证质量
        transforms.Resize(dest_image_size, interpolation=InterpolationMode.BICUBIC),  # BICUBIC 就是告诉电脑用"高质量模式"来缩放图片，让图片看起来更清晰自然。
        # 步骤2：将图像转为张量，像素值从 0-255 变为 0-1
        transforms.ToTensor(),
        # 步骤3：归一化，像素值从 0-1 变为 -1到1
        normalize
    ])

    return transform


def build_data_set(dest_image_size, data):
    """
    构建图像分类数据集
    参数: 
        dest_image_size - 目标图像尺寸
        data - 数据集根目录路径（子文件夹为类别名）
    返回: ImageFolder 数据集对象
    """
    transform = build_transform(dest_image_size)
    dataset = datasets.ImageFolder(data, transform=transform, target_transform=None)
    return dataset
