# -*- coding: utf-8 -*-
"""
基于 LSTM 的情感分析完整实现
使用 IMDB 数据集进行二分类任务（正面/负面情感）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
import os
import urllib.request
import tarfile


# ==================== 0. 数据集下载和加载 ====================

def download_imdb(data_dir='./data'):
    """下载 IMDB 数据集"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    tar_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')
    extract_path = os.path.join(data_dir, 'aclImdb')
    
    if os.path.exists(extract_path):
        print("数据集已存在，跳过下载")
        return extract_path
    
    print(f"正在下载 IMDB 数据集...")
    try:
        urllib.request.urlretrieve(url, tar_path)
        print("下载完成，正在解压...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        os.remove(tar_path)
        print("解压完成！")
        return extract_path
    except Exception as e:
        print(f"下载失败: {e}")
        print("使用示例数据代替...")
        return None


class IMDBDataset(Dataset):
    """自定义 IMDB 数据集"""
    
    def __init__(self, data_dir, split='train', use_sample=False):
        self.data = []
        
        if use_sample or data_dir is None:
            # 使用示例数据
            if split == 'train':
                self.data = [
                    ("This movie is great! I love it.", "pos"),
                    ("Terrible film, waste of time.", "neg"),
                    ("Amazing performance by the actors.", "pos"),
                    ("Boring and predictable plot.", "neg"),
                    ("Excellent cinematography and story.", "pos"),
                    ("Very disappointing, would not recommend.", "neg"),
                    ("A masterpiece! Must watch.", "pos"),
                    ("Poor acting and bad script.", "neg"),
                    ("Fantastic movie, highly recommended!", "pos"),
                    ("Awful experience, total disaster.", "neg"),
                ] * 250  # 2500 samples
            else:
                self.data = [
                    ("Good movie, enjoyed it.", "pos"),
                    ("Not good at all.", "neg"),
                    ("Best film I've seen.", "pos"),
                    ("Completely terrible.", "neg"),
                ] * 125  # 500 samples
        else:
            # 从文件加载
            for label in ['pos', 'neg']:
                dir_path = os.path.join(data_dir, split, label)
                if os.path.exists(dir_path):
                    for filename in os.listdir(dir_path):
                        if filename.endswith('.txt'):
                            with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                                text = f.read()
                                self.data.append((text, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def simple_tokenizer(text):
    """简单的分词器"""
    return text.lower().replace('.', ' .').replace('!', ' !').replace('?', ' ?').replace(',', ' ,').split()


# ==================== 1. 数据预处理 ====================

def yield_tokens(data_iter, tokenizer): #从数据集中提取所有的词（token），用于构建词汇表
    """从数据迭代器中提取 token"""
    for label, text in data_iter:  #data_iter: IMDB 数据集的迭代器，每次返回 (label, text) 对
        yield tokenizer(text) #yield: Python 生成器关键字，每次返回一个结果，但不会结束函数

#这个函数就像建立一个词典索引，把所有重要的词（出现频率 ≥ 5）都编号，方便后续模型处理。就像给每个学生分配学号一样。
def build_vocabulary(train_data, tokenizer, min_freq=5):   #min_freq=5: 最小词频，只保留出现次数 ≥ 5 的词。出现 ≥ 5 次的词 → 加入词汇表
    """构建词汇表"""
    word_count = {}
    
    # 统计词频
    for text, _ in train_data:
        tokens = tokenizer(text)
        for token in tokens:
            word_count[token] = word_count.get(token, 0) + 1
    
    # 构建词汇表
    vocab = {'<unk>': 0, '<pad>': 1}
    idx = 2
    for word, count in word_count.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    # 创建默认值字典（返回 <unk> 索引）
    class VocabDict(dict):
        def __missing__(self, key):
            return self['<unk>']
    
    return VocabDict(vocab)

#将一批原始文本数据转换成模型可以处理的张量格式，并且将文本序列长度统一，方便后续的训练。
def collate_batch(batch, tokenizer, vocab, device):
    """
    将一个 batch 的数据处理成模型需要的格式
    返回：文本的索引序列、序列长度、标签
    """
    label_list, text_list, length_list = [], [], []   
    
    for label, text in batch:
        # 标签处理：pos -> 1, neg -> 0
        label_list.append(1 if label == 'pos' else 0)
        
        # 文本处理：分词 -> 转索引
        tokens = tokenizer(text)
        text_indices = [vocab[token] for token in tokens]
        text_list.append(text_indices)
        length_list.append(len(text_indices))
        """
        # s上面的代码举个例子：假设 batch 有 3 条数据
        batch = [
            ('pos', "This movie is great"),
            ('neg', "Bad film"),
            ('pos', "I love it")
        ]
        # 处理后：
        label_list = [1, 0, 1]  # pos->1, neg->0

        text_list = [
            [2, 4, 3, 5],      # "This movie is great"
            [10, 6],           # "Bad film"
            [8, 7, 9]          # "I love it"
        ]
        length_list = [4, 2, 3]  # 每个句子的长度
        """
    
    # 转换为张量
    label_list = torch.tensor(label_list, dtype=torch.long)
    length_list = torch.tensor(length_list, dtype=torch.long)
    
    # 对文本进行填充，使一个 batch 内的序列长度相同
    # 按长度降序排列（pack_padded_sequence 要求）
    length_list, sorted_idx = length_list.sort(descending=True)
    label_list = label_list[sorted_idx]
    text_list = [text_list[i] for i in sorted_idx]
    """ 
    举个例子 排序前：
    位置0: text=[2,4,3,5]    label=1  length=4  ← 最长
    位置1: text=[10,6]       label=0  length=2  ← 最短  
    位置2: text=[8,7,9]      label=1  length=3  ← 中等
    排序后：
    位置0: text=[2,4,3,5]    label=1  length=4  ← 最长
    位置1: text=[8,7,9]      label=1  length=3  ← 中等
    位置2: text=[10,6]       label=0  length=2  ← 最短
    """

    # 将不同长度的序列填充到相同长度，然后转换为张量并移动到设备
    padded_text_list = []
    max_len = length_list[0].item()
    for text in text_list:
        padded = text + [vocab['<pad>']] * (max_len - len(text))
        padded_text_list.append(padded)
    
    text_list = torch.tensor(padded_text_list, dtype=torch.long)
    """
    举例 填充前：
    text_list[0] = [2, 4, 3, 5]         ← 长度 4
    text_list[1] = [8, 7, 9]            ← 长度 3 (短1个)
    text_list[2] = [10, 6]              ← 长度 2 (短2个)
    ❌ 无法组成矩形张量！
    填充后：                       #说明：<pad> 这个特殊标记在词汇表中的索引是1
    padded_text_list[0] = [2, 4, 3, 5]      ← 长度 4
    padded_text_list[1] = [8, 7, 9, 1]      ← 长度 4 (填充1个)
    padded_text_list[2] = [10, 6, 1, 1]     ← 长度 4 (填充2个)
    ✅ 可以组成矩形张量 [3, 4]！
    """
    return text_list.to(device), length_list.to(device), label_list.to(device)


# ==================== 2. 模型定义 ====================

class LSTM(nn.Module):
    """
    LSTM 情感分析模型
    使用双向 LSTM + Dropout + 全连接层
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout_rate, pad_idx):
        super(LSTM, self).__init__()
        
        # Embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            embedding_dim,                                  # 词嵌入维度（如 300）
            hidden_dim,                                     # LSTM 隐藏层维度（如 300）
            num_layers=n_layers,                            # LSTM 层数（如 2）
            bidirectional=bidirectional,                    # 是否使用双向 LSTM（如 True）
            dropout=dropout_rate if n_layers > 1 else 0,    # Dropout 比率（如 0.5）
            batch_first=True                                # 是否使用 batch_first（如 True）
        )
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接层
        # 如果是双向 LSTM，hidden_dim 要乘以 2
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
    
    def forward(self, text, text_lengths):
        """
        前向传播
        text: [batch_size, seq_len]
        text_lengths: [batch_size]
        """
        # Embedding
        # embedded: [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(text))  #为什么在 Embedding 后加 Dropout？防止过拟合。
        
        # 打包序列（提高效率，避免对 padding 部分进行计算）
        packed_embedded = pack_padded_sequence(
            embedded,              # 输入：填充后的词向量
            text_lengths.cpu(),    # 输入：每个序列的长度
            batch_first=True,      # 输入：是否使用 batch_first
            enforce_sorted=True    # 输入：是否对序列进行排序
        )
        
        # LSTM
        # packed_output: 打包的输出
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        # cell: [num_layers * num_directions, batch_size, hidden_dim]
        packed_output, (hidden, cell) = self.lstm(packed_embedded)    #packed_output - 每个时间步的输出，hidden - 最后的隐藏状态，cell - 细胞状态
        
        # 解包（本例中不使用 output，直接使用 hidden）
        # output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        
        # 处理双向 LSTM 的隐藏状态
        # 取最后一层的正向和反向的隐藏状态进行拼接
        if self.lstm.bidirectional:
            # hidden[-2]: 最后一层正向的隐藏状态
            # hidden[-1]: 最后一层反向的隐藏状态
            hidden = self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=-1))
        else:
            # hidden[-1]: 最后一层的隐藏状态
            hidden = self.dropout(hidden[-1])
        
        # 全连接层
        # output: [batch_size, output_dim]
        output = self.fc(hidden)
        
        return output


# ==================== 3. 训练和评估函数 ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for text, text_lengths, labels in dataloader:
        # 前向传播
        predictions = model(text, text_lengths)
        
        # 计算损失
        loss = criterion(predictions, labels)
        
        # 计算准确率
        predicted_labels = predictions.argmax(dim=1)
        acc = (predicted_labels == labels).float().mean()
        '''
        predictions:
        ┌──────────────┐
        │  2.3   -1.5  │ → argmax → 0 (2.3 > -1.5)
        │ -0.8    1.9  │ → argmax → 1 (1.9 > -0.8)
        │  1.2    0.3  │ → argmax → 0 (1.2 > 0.3)
        │  0.5    0.7  │ → argmax → 1 (0.7 > 0.5)
        └──────────────┘
            ↓ dim=1
        predicted_labels = [0, 1, 0, 1]
        '''
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for text, text_lengths, labels in dataloader:
            # 前向传播
            predictions = model(text, text_lengths)
            
            # 计算损失
            loss = criterion(predictions, labels)
            
            # 计算准确率
            predicted_labels = predictions.argmax(dim=1)
            acc = (predicted_labels == labels).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


def epoch_time(start_time, end_time):
    """计算时间"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# ==================== 4. 预测函数 ====================

def predict_sentiment(model, tokenizer, vocab, sentence, device):
    """
    预测单个句子的情感
    """
    model.eval()  #评估模式：Dropout 关闭：不再随机丢弃神经元，BatchNorm 使用训练时的统计量（如果有）
    
    # 分词并转换为索引
    tokens = tokenizer(sentence)  
    indices = [vocab[token] for token in tokens]
    length = torch.tensor([len(indices)], dtype=torch.long)
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # 添加 batch 维度
    
    # 转移到设备
    tensor = tensor.to(device)
    length = length.to(device)
    
    # 预测
    with torch.no_grad():    #禁用梯度计算：节省内存，加速计算，预测时不需要反向传播
        prediction = model(tensor, length)
        probabilities = torch.softmax(prediction, dim=1)  # 转换为概率
        predicted_class = prediction.argmax(dim=1).item()
    
    # 返回结果
    sentiment = 'pos' if predicted_class == 1 else 'neg'
    confidence = probabilities[0][predicted_class].item()
    
    return sentiment, confidence


# ==================== 5. 主函数 ====================

def main():
    """主函数"""
    
    print("=" * 60)
    print("基于 LSTM 的情感分析实战")
    print("=" * 60)
    
    # 设置随机种子
    SEED = 1234
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # ==================== 数据加载 ====================
    print("\n" + "=" * 60)
    print("1. 加载数据集...")
    print("=" * 60)
    
    # 获取分词器
    tokenizer = simple_tokenizer
    
    # 下载/加载 IMDB 数据集
    print("准备 IMDB 数据集...")
    
    # 检查是否需要下载（可以设置环境变量 USE_SAMPLE_DATA=1 直接使用示例数据）
    import os as os_module
    force_sample = os_module.environ.get('USE_SAMPLE_DATA', '0') == '1'
    
    if force_sample:
        print("📝 使用示例数据模式（设置了 USE_SAMPLE_DATA=1）")
        data_dir = None
        use_sample = True
    else:
        data_dir = download_imdb('./data')
        use_sample = (data_dir is None)
    
    if use_sample:
        print("⚠️ 使用示例数据（2500 训练样本，500 测试样本）")
        print("提示：要使用完整数据集，请等待下载完成（约84MB）")
    else:
        print("✓ 使用完整 IMDB 数据集（25000 训练样本，25000 测试样本）")
    
    # 加载数据集
    train_dataset = IMDBDataset(data_dir, split='train', use_sample=use_sample)
    test_dataset = IMDBDataset(data_dir, split='test', use_sample=use_sample)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 构建词汇表
    print("\n构建词汇表...")
    vocab = build_vocabulary(train_dataset, tokenizer, min_freq=2 if use_sample else 5)
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # ==================== 创建数据加载器 ====================
    print("\n" + "=" * 60)
    print("2. 创建数据加载器...")
    print("=" * 60)
    
    BATCH_SIZE = 64
    
    # 创建 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, vocab, device)
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, vocab, device)
    )
    
    print(f"训练集批次数: {len(train_dataloader)}")
    print(f"测试集批次数: {len(test_dataloader)}")
    
    # ==================== 模型初始化 ====================
    print("\n" + "=" * 60)
    print("3. 初始化模型...")
    print("=" * 60)
    
    # 模型超参数
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300
    OUTPUT_DIM = 2
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT_RATE = 0.5
    PAD_IDX = vocab['<pad>']
    
    # 创建模型
    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout_rate=DROPOUT_RATE,
        pad_idx=PAD_IDX
    )
    
    # 移动到设备
    model = model.to(device)
    
    # 打印模型信息
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数量: {count_parameters(model):,}")
    print(f"\n模型结构:\n{model}")
    
    # ==================== 训练设置 ====================
    print("\n" + "=" * 60)
    print("4. 训练设置...")
    print("=" * 60)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    # 训练轮数
    N_EPOCHS = 5
    
    print(f"优化器: Adam (lr=1e-3)")
    print(f"损失函数: CrossEntropyLoss")
    print(f"训练轮数: {N_EPOCHS}")
    
    # ==================== 训练模型 ====================
    print("\n" + "=" * 60)
    print("5. 开始训练...")
    print("=" * 60)
    
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device)
        
        # 验证（使用测试集）
        valid_loss, valid_acc = evaluate(model, test_dataloader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './lstm_best.pt')
        
        # 打印训练信息
        print(f'\nEpoch: {epoch+1:02} | 时间: {epoch_mins}m {epoch_secs}s')
        print(f'\t训练损失: {train_loss:.3f} | 训练准确率: {train_acc*100:.2f}%')
        print(f'\t验证损失: {valid_loss:.3f} | 验证准确率: {valid_acc*100:.2f}%')
    
    # ==================== 最终评估 ====================
    print("\n" + "=" * 60)
    print("6. 最终评估...")
    print("=" * 60)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('./lstm_best.pt'))
    
    # 在测试集上评估
    test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)
    
    print(f'\n测试集损失: {test_loss:.3f}')
    print(f'测试集准确率: {test_acc*100:.2f}%')
    
    # ==================== 预测示例 ====================
    print("\n" + "=" * 60)
    print("7. 预测示例...")
    print("=" * 60)
    
    # 测试一些句子
    test_sentences = [
        "This film is terrible!",
        "This film is great!",
        "I love this movie, it's amazing!",
        "I hate this movie, it's boring.",
        "The acting was superb and the plot was engaging.",
        "Waste of time and money."
    ]
    
    print()
    for sentence in test_sentences:
        sentiment, confidence = predict_sentiment(model, tokenizer, vocab, sentence, device)
        print(f"句子: {sentence}")
        print(f"预测: {sentiment} (置信度: {confidence:.4f})")
        print()
    
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

