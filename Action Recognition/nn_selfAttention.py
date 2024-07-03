"""
6.数据预处理 并 创建标签label和特征feature
Preprocess data and create Labels and features
7.创建并训练LSTM NN
build and train LSTM nerual network
8.预测
make predictions
9.保存模型
save weights
10. 使用混淆矩阵与精度进行评估
evaluation using Confusion matrix and accuracy
"""
import os

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ——————————————————————————————————————————————————————————
# 数据地址


class myDataset(Dataset):
    def __init__(self, data_path, model="train"):
        # 数据地址
        self.data_path = data_path
        self.gesture_size = 20
        # 模式
        self.model = model
        # 手部数据
        self.hands_data = []
        # 手势标签
        self.gesture_label = []
        # 手势的种类
        self.gestures_len = os.listdir(self.data_path)
        label_map = {label: num for num, label in enumerate(self.gestures_len)}
        # 写入
        for num, action in enumerate(self.gestures_len):
            sequences = os.listdir(self.data_path + '/' + str(self.gestures_len[num]))  # 获得下一级目录的各个文件夹
            # print(len(sequences))  # 300
            for sequence in range(len(sequences)):
                sequnece_length = os.listdir(
                    self.data_path + '/' + str(self.gestures_len[num] + '/' + str(sequence)))  # 获得下下一级目录的各个文件夹
                x = []  # x由 20 x .npy  组合成
                for frame_key in range(len(sequnece_length)):
                    res = np.load(os.path.join(self.data_path, action, str(sequence), '{}.npy'.format(frame_key * 3)))
                    numpy.array(res)
                    x.append(res)
                self.hands_data.append(x)
                self.gesture_label.append(label_map[action])
        self.hands_data = torch.FloatTensor(numpy.array(self.hands_data))
        # self.gesture_label = torch.FloatTensor(self.gesture_label)

    def __len__(self):
        return len(self.hands_data)

    def __getitem__(self, index):
        if self.model == "train":
            return self.hands_data[index], self.gesture_label[index]
        else:
            return [self.hands_data[index], self.gesture_label[index]]

    def get_gesture_number(self):
        return self.gestures_len


from torch.nn.utils.rnn import pad_sequence


# 用于记录样本分布


def collate_batch(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    hands_data, gesture_label = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    # 因为我们一批一批地训练模型，所以我们需要在同一批中填充特征以使其长度相同。
    # pad_sequence()沿新维度堆叠张量列表
    # 在使用pytorch训练模型的时候，一般采用batch的形式同时处理多个样本序列，而同一batch中时序信息的的长度是不同的，
    # 这样就无法传入RNN，LSTM，GRU这样的模型中进行处理。一个常用的做法是按照一个指定的长度(或者按照batch中最长的序列长度)
    # 对batch中的序列进行填充(padding)或者截断(truncate)，这样就会导致一些较短的序列中会有很多的填充符。
    #
    # batch_first参数表示tensor堆叠后第一个参数是否时批次大小，默认是否，
    # 填充元素的值。
    hands_data = pad_sequence(hands_data, batch_first=True)
    # print("一个批次的x数据形式为:{}".format(np.array(hands_data).shape))
    # hands_data: (batch size, length, 40)    (batch_size,20,63)
    return hands_data, torch.FloatTensor(gesture_label).long()


def get_dataloader(data_path, batch_size, n_workers=0, split_key=0.8, model='train'):
    """
    Generate dataloader
    :param model: 数据加载时的模式
    :param data_path: 数据地址
    :param batch_size: 批大小
    :param n_workers: 用于数据的子进程数量装载0表示数据将在主进程中加载。（默认值：0）
    :param split_key: 训练集占比
    :return:训练dataloader，验证dataloader，手势种类
    """
    # 创建dataset
    hands_dataset = myDataset(data_path, model=model)

    gesture_number = hands_dataset.get_gesture_number()
    # 将数据集分割,将数据集拆分为训练数据集和验证数据集
    train_len = int(split_key * len(hands_dataset))
    lengths = [train_len, len(hands_dataset) - train_len]
    # random_split()用于随机采样
    train_dataset1, valid_dataset = random_split(hands_dataset, lengths)

    train_len2 = int(0.9 * len(train_dataset1))
    lengths2 = [train_len2, len(train_dataset1) - train_len2]
    train_dataset, test_dataset = random_split(train_dataset1, lengths2)

    # 训练集的dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=n_workers,
                              pin_memory=True,
                              collate_fn=collate_batch,
                              )
    # 验证集的dataloader
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              num_workers=n_workers,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_batch,
                              )
    test_loader = DataLoader(test_dataset,
                             batch_size=120,
                             num_workers=n_workers,
                             drop_last=True,
                             pin_memory=True,
                             collate_fn=collate_batch,
                             )

    return train_loader, valid_loader, test_loader, gesture_number


# ——————————————————————————————————————————————————————————————————————————————————
# 构建网络
import torch.nn as nn


class Self_Attention(nn.Module):
    # d_model：输入的预期特征数,这个特征值属于中间参数和输入输出的特征大小均没有关系
    def __init__(self, d_model=80, n_gestures=5, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        # 将特征的维度从输入的维度投影到d_model中。
        self.prenet = nn.Linear(in_features=63, out_features=d_model)
        #
        # TransformerEncoderLayer由自适应和前馈网络组成。该标准编码器层基于论文“注意力就是你所需要的”
        # 参数：
        # d_model：输入中预期特征的数量,
        # nhead：多头注意力模型中的Multihead的数，
        # dim_feedforward：前馈网络模型的维度（默认值 = 2048）。
        # dropout：dropout值（默认值 = 0.1）。
        # activation：中间层的激活函数，可以是字符串（“relu”或“gelu”）或一元可调用函数。默认值：relu
        # layer_norm_eps：层规范化组件中的eps值（默认值 = 1e-5）。
        # batch_first：如果“True”，则输入和输出张量提供为（batch、seq、feature）。默认值：``False ``（序列、批次、功能）。
        # norm_first：如果“True”，则在注意和前馈操作之前分别执行层规范。否则，它会在之后完成。默认值：``False ``（之后）。

        # Examples::
        # >> > encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # >> > src = torch.rand(10, 32, 512)
        # >> > out = encoder_layer(src)

        # Alternatively, when
        # ``batch_first`` is ``True``:
        # >> > encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        # >> > src = torch.rand(32, 10, 512)
        # >> > out = encoder_layer(src)
        # 更好的网络——————————————————————————————————————————
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        # ————————————————————————————————————————————————————
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2, dropout=dropout
        )
        # 注意力层的数量
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the dimension of features from d_model into speaker nums.
        # 将特征维度从d_model连接到手势种类中。
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=24),
            nn.ReLU(),
            nn.Linear(in_features=24, out_features=n_gestures)
        )

    def forward(self, hands_data):
        """
    args:
      dataloader: (batch size, data_length, 20)
    return:
      out: (batch size, gesture_types)
    """
        # input:(batch_size,20,63)
        # 将特征的维度从输入的维度投影到d_model中
        out = self.prenet(hands_data)
        # out: (batch size, length, d_model)
        #
        # permute()将tensor的维度换位。原来顺序为（0，1，2）现在为（1，0，2）
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # 编码器层期望特征的形状为（序列、批次大小、d_model（特征数量））。
        out = self.encoder(out)
        # transpose()将tensor的维度换位回来
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)
        # out: (batch, gesture_types)
        out = self.pred_layer(stats)
        return out


# ______________________________________________________________________________________
# 训练网络
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
# Python处理比较耗时操作的时候，为了便于观察处理进度，这时候就需要通过进度条将处理情况进行可视化展示，以便我们能够及时了解情况
from tqdm import tqdm
# 数据可视化
from tensorboardX import SummaryWriter


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1):
    """
  Create a schedule with a learning rate that decreases following the values of the cosine function between the
  initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
  initial lr set in the optimizer.

  Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

  Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
  """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion):
    """Forward a batch through the model."""

    hands_data, gesture_label = batch
    # print(gesture_label)

    outs = model(hands_data)
    # print(outs.argmax(1))
    loss = criterion(outs, gesture_label)

    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == gesture_label).float())

    return loss, accuracy


def valid(dataloader, model, criterion):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            accuracy=f"{running_accuracy / (i + 1):.2f}",
        )
    pbar.close()
    model.train()

    return running_loss / len(dataloader), running_accuracy / len(dataloader)


def train(save_path, train_loader, valid_loader,
          warmup_steps, total_steps, valid_steps, save_steps, my_model):
    """训练"""

    print("...start train...")
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!", flush=True)
    # 初始化神经网络
    nn_model = my_model
    # 初始化训练记录模块
    writer = SummaryWriter()
    # 初始化参数
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(nn_model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!", flush=True)
    # 初始化记录
    loss_record = {'train_loss': [], 'validation_loss': []}
    accuracy_record = {'train_accuracy': [], 'validation_accuracy': []}
    best_accuracy = -1.0
    best_state_dict = None
    # 可视化python训练过程
    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        nn_model.train()
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        # 前向计算
        loss, accuracy = model_fn(batch, nn_model, criterion)

        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # loss值记录
        loss_record['train_loss'].append(batch_loss)
        # accuracy值记录
        accuracy_record['train_accuracy'].append(batch_accuracy)
        # 记录
        writer.add_scalar('train_loss', batch_loss, global_step=step)
        writer.add_scalar('train_accuracy', batch_accuracy, global_step=step)

        # 反向传播更新参数
        # Updata model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()
            nn_model.eval()
            # print("验证预测")
            valid_loss, valid_accuracy = valid(valid_loader, nn_model, criterion)
            loss_record['validation_loss'].append(valid_loss)
            accuracy_record['validation_accuracy'].append(valid_accuracy)
            writer.add_scalar('validation_loss', valid_loss, global_step=step)
            writer.add_scalar('validation_accuracy', valid_accuracy, global_step=step)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = nn_model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()
    return loss_record, accuracy_record


from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve, accuracy_score
from torchinfo import summary
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='State transition matrix'):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='blue')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
    #     plt.text(j, i, num,
    #              verticalalignment='center',
    #              horizontalalignment="center",
    #              color="white" if num > thresh else "black")

    plt.ylabel('Self patt')
    plt.xlabel('Transition patt')
    plt.tight_layout()
    plt.savefig('res/method_2.png', transparent=True, dpi=800)
    plt.show()


def predict(model, testingDataloader):
    model.eval()
    preds = []
    for x in testingDataloader:
        with torch.no_grad():
            pred = model(x).squeeze(1)
            preds.append(pred.detach())
    preds = torch.cat(preds, dim=0).numpy()

    return preds


# 训练及保存模型
if __name__ == "__main__":
    config = {'save_path': "recognize_model_300_Atth.pth",
              'warmup_steps': 100,
              'total_steps': 1000,
              'valida_steps': 10,
              'save_steps': 20,
              'data_path': 'gesture_Data',
              'batch_size': 15,
              'gesture_types': 5
              }

    # print("数据加载开始....")
    # dataset = myDataset(data_path=config['data_path'])
    # x_sequence0, label0 = dataset[0]
    # print("输入值为x序列，举例：{}".format(x_sequence0))
    # print("标签值y，举例：{}".format(label0))
    # print("输入数据格式为：{}，样本大小为：{}".format(np.array(x_sequence0).shape, len(dataset)))  # (20,63),1500

    selfAttentionModel = Self_Attention(d_model=80, n_gestures=config['gesture_types'], dropout=float(0.2))
    train_loader, valid_loader, test_loader, gesture_types = get_dataloader(data_path=config['data_path'],
                                                                            batch_size=config['batch_size'])
    print("....Data loading was finished")
    is_train = False
    if is_train:

        Loss_record, Valid_record = train(config['save_path'],
                                          train_loader, valid_loader,
                                          config['warmup_steps'],
                                          config['total_steps'],
                                          config['valida_steps'],
                                          config['save_steps'],
                                          my_model=selfAttentionModel)
    else:
        selfAttentionModel.load_state_dict(torch.load(config['save_path']))
        # input(batch_size, 20, 63)
        summary(selfAttentionModel, [150, 20, 63])
        print("predict：")
        path = 'gesture_Data'
        actions = os.listdir(path)
        # print(actions)  [['pinch', 'press', 'insert', 'screw', 'prod']]
        label_map = {label: num for num, label in enumerate(actions)}
        print(actions[0])

        selfAttentionModel.eval()
        for x_yTure in test_loader:
            # print(x_yTure)
            x = x_yTure[0]
            # print(x)
            yTrue = x_yTure[1].tolist()
            true_label = []
            for y in yTrue:
                true_label.append(actions[y])
            print(true_label)
            yhat = selfAttentionModel(x)
            # print(np.array(x).shape)

            # print(yhat)
            yhat = yhat.detach().numpy()
            # print(yhat.shape)
            pre_label = []
            y_score=[]
            for i in np.array(yhat):
                yPre = np.argmax(i)
                # print(yPre)
                pre_label.append(actions[yPre])
            print(pre_label)
            # # 真实值
            # # 混肴矩阵
            multi_confusion = multilabel_confusion_matrix(true_label, pre_label, labels=actions)
            print("multi_confusion：{}".format(multi_confusion))
            # # 正确分数
            acc_score = accuracy_score(true_label, pre_label)
            print("accuracy_score：{}".format(acc_score))
            # precision = precision_score(true_label, pre_label, labels=actions)
            # print("precision_score：{}".format(precision))
            # sensitivity = recall_score(true_label, pre_label, labels=actions)
            # print("sensitivity_score：{}".format(sensitivity))
            confusion_ matrix(true_label, pre_label)

