import torchtext
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# 生成训练数据
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import tqdm
import sys
import numpy as np

torchtext.disable_torchtext_deprecation_warning()
# 读取数据
train_iter = load_dataset(path="imdb", cache_dir="./data", split="train")

# 创建分词器
tokenizer = get_tokenizer("basic_english")
print(tokenizer("here is the an example!"))


# 构建词汇表
def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item['text'])
# 根据分词器的输出构建词汇表，<pad> 和 <unk> 被设置为特殊标记
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
# 当词汇表中找不到某个词时，使用 <unk> 作为默认索引
vocab.set_default_index(vocab["<unk>"])
print(vocab(tokenizer('here is the an example <pad> <pad>')))

# 将文本转换为词汇表索引的序列
text_pipeline = lambda x: vocab(tokenizer(x))
# 将标签 "pos" 映射为 1，"neg" 映射为 0
label_pipeline = lambda x: 1 if x == 'pos' else 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 批次数据的处理
def collate_batch(batch):
    max_length = 256
    pad = text_pipeline('<pad>')
    label_list, text_list, length_list = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = text_pipeline(_text)[:max_length]
        length_list.append(len(processed_text))
        text_list.append((processed_text + pad * max_length)[:max_length])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.tensor(text_list, dtype=torch.int64)
    length_list = torch.tensor(length_list, dtype=torch.int64)
    return label_list.to(device), text_list.to(device), length_list.to(device)


# 使用 to_map_style_dataset 函数将迭代器转化为 Dataset 类型；
train_dataset = to_map_style_dataset(train_iter)
# 使用 random_split 函数对 Dataset 进行划分，其中 95% 作为训练集，5% 作为验证集；
num_train = int(len(train_dataset) * 0.95)

# 拆分训练集 和 验证集
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
# 生成训练集的 DataLoader；
train_dataloader = DataLoader(split_train_, batch_size=8, shuffle=True, collate_fn=collate_batch)
# 生成验证集的 DataLoader
valid_dataloader = DataLoader(split_valid_, batch_size=8, shuffle=False, collate_fn=collate_batch)


# 定义模型
class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate,
                 pad_index=0):
        super().__init__()
        # Embedding：将词汇表中的词映射到嵌入空间
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        # LSTM 层，用于处理变长的序列数据
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                                  dropout=dropout_rate, batch_first=True)
        # 全连接层，输出预测结果
        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        # 防止过拟合
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True,
                                                                  enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_length = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        prediction = self.fc(hidden)
        return prediction


# 实例化模型
vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 300
output_dim = 2
n_layers = 2
bidirectional = True
dropout_rate = 0.5

model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate)
model = model.to(device)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)
# 优化方法
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        (label, ids, length) = batch
        label = label.to(device)
        ids = ids.to(device)
        length = length.to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)  # loss计算
        accuracy = get_accuracy(prediction, label)
        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return epoch_losses, epoch_accs


def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            (label, ids, length) = batch
            label = label.to(device)
            ids = ids.to(device)
            length = length.to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)  # loss计算
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return epoch_losses, epoch_accs

# 准确率评估
def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


n_epochs = 10
best_valid_loss = float('inf')

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

for epoch in range(n_epochs):
    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
    valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)
    train_losses.extend(train_loss)
    train_accs.extend(train_acc)
    valid_losses.extend(valid_loss)
    valid_accs.extend(valid_acc)
    epoch_train_loss = np.mean(train_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_valid_loss = np.mean(valid_loss)
    epoch_valid_acc = np.mean(valid_acc)
    if epoch_valid_loss < best_valid_loss:
        best_valid_loss = epoch_valid_loss
        torch.save(model.state_dict(), 'lstm.pt')
    print(f'epoch: {epoch + 1}')
    print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
    print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')
