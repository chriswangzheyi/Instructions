import torch.nn as nn


class DiceLoss(nn.Module):
    # Dice损失函数，用于图像分割

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0  # 防止除以0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()  #assert - 断言，如果不相等就报错停止
        # 展平成一维
        y_pred = y_pred.contiguous().view(-1)   #contiguous()确保数据在内存中连续存储
        y_true = y_true.contiguous().view(-1)   #view(-1)将张量展平成一维

        # 计算交集：相乘后只有都为1时结果才为1，求和得到重叠像素数
        # 例：y_pred=[1,0,0,1], y_true=[1,1,0,0] → 相乘=[1,0,0,0] → sum=1
        # intersection 的值就是预测对了多少个像素
        intersection = (y_pred * y_true).sum() 

        # Dice系数 = 2*交集 / 总和，如果预测和真实完全一样，Dice = 1（完美），如果完全不重叠，Dice = 0（很差）
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc  # 损失 = 1 - Dice系数， 因为Dice系数越大越好，loss越小越好
