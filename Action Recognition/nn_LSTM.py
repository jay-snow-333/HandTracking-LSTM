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
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import os

# ——————————————————————————————————————————————————————————
# 数据地址
path = 'gesture_Data'
# 模型名称
model_name = 'recognize_model_LSTM.h5'
Batch_size = 30
Epochs = 300
# 读取手势数据集文件夹
actions = os.listdir(path)
# 遍历文件夹下的子文件名称
# print(actions)  [['pinch', 'press', 'insert', 'screw', 'prod']]
label_map = {label: num for num, label in enumerate(actions)}
print(actions[0])

# x和label
# 读取数据
x_sequences, labels = [], []
for num, action in enumerate(actions):
    sequences = os.listdir(path + '/' + str(actions[num]))  # 获得下一级目录的各个文件夹
    # print(len(sequences))  # 300
    for sequence in range(len(sequences)):
        sequnece_length = os.listdir(path + '/' + str(actions[num] + '/' + str(sequence)))  # 获得下下一级目录的各个文件夹
        x = []  # x由 20 x .npy  组合成
        # 这个20指的是手势数据的长度
        if len(sequnece_length) != 20:
            print(len(sequnece_length))
            print("{}/{}/{}".format(path, actions[num], sequence))
        for frame_key in range(len(sequnece_length)):
            res = np.load(os.path.join(path, action, str(sequence), '{}.npy'.format(frame_key * 3)))
            # 63是每帧数据的手势特征点个数
            if len(res) != 63:
                print(len(res))
            x.append(res)
        x_sequences.append(x)
        labels.append(label_map[action])
# 输入为x_seq
# print("输入值为x序列，举例：{}".format(x_sequences[0]))
print("标签值y，举例：{}".format(labels[0]))
print(np.array(x_sequences).shape)  # (1500,20,63)   x的形式为
print(len(labels))  # 300x5            label的形式为
x_sequences = np.array(x_sequences)
print("标签变更为one-hot vector...")
y_values = to_categorical(labels).astype(int)
print("最终x和y的shape为：")
print(x_sequences.shape)
print(y_values.shape)
# ------------------------------------数据读取完成--------


# 分割数据
print("进行数据分割...")
x_train, x_test, y_train, y_test = train_test_split(x_sequences, y_values, test_size=0.1, shuffle=True)
print("训练集样本大小为：{}".format(len(y_train)))
print("测试集样本大小为：{}".format(len(y_test)))
# ——————————————————————————————————————————————————————————————————————————————————
# 7.创建并训练LSTM NN
# 这个是神经网络模型构建的方法
from keras.models import Sequential
# 导入Lstm和全连接层
from keras.layers import LSTM, Dense
# 记录板，保存至Logs文件夹下
from keras.callbacks import TensorBoard
# 导入优化函数
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

# 记录训练过程,可以在一个网站上下载此记录
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# # 创建神经网络
model = Sequential()

model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(20, 63)))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
# 是返回输出序列中的最后一个输出，还是返回完整序列。return_sequences=False
# 此处为false才能使得输入和输出的sequence不一致，即输入的shape和输出的shape是可以改变的
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))  # output = (None, 20, 5)  y_train.shape =  (85, 30, 3)

print(model.summary())  # 打印神经网络结构

import matplotlib.pyplot as plt

# # 定义损失函数和优化函数
Optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics='categorical_accuracy')
#
# # 定义批大小和迭代次数
batch_size = Batch_size
epochs = Epochs
is_train = False
if is_train:
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=tb_callback,
                        validation_split=0.2,
                        validation_batch_size=batch_size)
    model.save(model_name)

    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.show()
# 打开保存的logs文件的方法：
# 使用Anaconda Prompt（tensorflow环境）进入到保存的目录中
# 运行命令  tensorboard --logdir=.    即可
# ____________________________________________________________________________________
# 绘制结果


# _______________________________________________________________________
# 9.保存模型
# save weights
# model.save('action.h5')
# _____________________________________________________________________
# 8.预测
# make predictions
# 预测原理
# ress = [0.7, 0.2, 0.1]
# print(np.argmax(ress))
# print(actions[np.argmax(ress)])
# 进行测试
print("进行预测...")
model.load_weights(model_name)
# print("测试值为:{}".format(x_test[4]))
print("测试结果真实为：{}".format(y_test[0]))
res = model.predict(x_test)
print("测试值预测结果为：{}".format(actions[np.argmax(res[0])]))

# ____________________________________________________________________________
# 10.混淆矩阵和准确性
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

yhat = model.predict(x_test)
# 真实值
yTure = np.argmax(y_test, axis=1).tolist()
# 预测值
yPredict = np.argmax(yhat, axis=1).tolist()
# 混肴矩阵
multi_confusion = multilabel_confusion_matrix(yTure, yPredict)
print("混肴矩阵结果：{}".format(multi_confusion))

# 正确分数
acc_score = accuracy_score(yTure, yPredict)
print("正确率分数：{}".format(acc_score))