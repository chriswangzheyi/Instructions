# Train
from efficientnet import EfficientNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import argparse
from dataset import build_data_set
import os



def train(train_loader, model, criterion, optimizer, epoch, args, device):
    # 切换到训练模式（启用 Dropout 和 BatchNorm 等训练特性）
    model.train()

    # 遍历数据加载器，每次取一批图片和标签
    for i, (images, target) in enumerate(train_loader):
        # 将数据移到 GPU（如果可用），加速计算
        images = images.to(device)
        target = target.to(device)
        print(images.shape)
        
        # 前向传播：把图片输入模型，得到预测结果
        output = model(images)
        # 计算损失：比较预测结果和真实标签，看预测有多"错"
        loss = criterion(output, target)
        print('Epoch ', epoch, loss)

        # 反向传播三步走：
        # 步骤1：清空上一次的梯度（防止累积）
        optimizer.zero_grad()
        # 步骤2：反向传播，计算梯度（每个参数该怎么调整）
        loss.backward()
        # 步骤3：根据梯度更新模型参数（让模型变得更准确）
        optimizer.step()


def main(args):
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
    
    # 设置分类类别数量（logo 和 others 两类）
    args.classes_num = 2
    
    # 创建模型：优先使用预训练模型（已在大量图片上训练过，效果更好）
    if args.pretrained:
        # 加载预训练模型，在此基础上微调
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=args.classes_num, advprop=args.advprop)
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        # 从零开始创建模型（需要更多数据和训练时间）
        print("=> creating model '{}'".format(args.arch))
        model = EfficientNet.from_name(args.arch, override_params={'num_classes': args.classes_num})
    
    # 将模型移到 GPU 上（如果可用），大幅提升训练速度
    model = model.to(device)

    # 定义损失函数：衡量模型预测有多"错"
    criterion = nn.CrossEntropyLoss()
    # 定义优化器：负责调整模型参数，让模型变得更准确
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 构建训练数据集
    train_dataset = build_data_set(args.image_size, args.train_data)
    # 创建数据加载器：分批次加载数据，每批10张图片，随机打乱顺序
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 训练循环：把所有数据训练多轮（默认10轮）
    for epoch in range(args.epochs):
        # 执行一轮训练
        train(train_loader, model, criterion, optimizer, epoch, args, device)
        # 每隔一定间隔保存模型（默认每轮都保存）
        if epoch % args.save_interval == 0:
            # 如果保存目录不存在，创建它
            if not os.path.exists(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            # 保存模型权重到文件，文件名包含轮次编号
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar.epoch_%s' % epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u"[+]----------------- 图像分类Sample -----------------[+]")
    parser.add_argument('--train-data', default='./data/train', dest='train_data', help='location of train data')
    parser.add_argument('--image-size', default=224, dest='image_size', type=int, help='size of input image')
    parser.add_argument('--batch-size', default=10, dest='batch_size', type=int, help='batch size')
    parser.add_argument('--workers', default=4, dest='num_workers', type=int, help='worders number of Dataloader')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr',  default=0.001,type=float, help='learning rate')
    parser.add_argument('--checkpoint-dir', default='./ckpts/', dest='checkpoint_dir', help='location of checkpoint')
    parser.add_argument('--save-interval', default=1, dest='save_interval', type=int, help='save interval')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay', help='weight decay')
    parser.add_argument('--arch', default='efficientnet-b0', help='arch type of EfficientNet')
    parser.add_argument('--pretrained', default=True, help='learning rate')
    parser.add_argument('--advprop', default=False, help='advprop')
    args = parser.parse_args()
    main(args)
