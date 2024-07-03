"""
1.使用meidapipe 收集关键点
collection keypoints from meidiapipe holistic
2.使用序列的LSTM层训练深层神经网络
train a  deep neural network with LSTM layers for Sequences
3.使用OpenCV执行实时手语检测
perform real time sign language detection using OpenCV
"""
import numpy as np

"""
1.下载并导入相关工具包
import and install Dependencies
2.使用MP获得keypoints
keypoints using MP Holistic
3.提取keypoints
extract Keypoint Values
4.为集合设置文件夹
setup Folder for Collection
5.收集Keypoints values 作为训练集和测试集
Collect keypoints Values for traing and testing
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

import matplotlib.pyplot as plt

hands = ["pinch", "press", "insert", "screw", "prod"]
# ['pinch', 'press', 'insert', 'screw', 'prod']
count1 = [45, 42, 39, 45, 44]
count2 = [45, 47, 43, 46, 45]

# 如果两句plt.bar()都写tick_label参数，则后面一句会覆盖前面一句，即展示在“b”柱下
x = np.arange(5)

plt.title("50 times test result")
plt.bar(x, count1, width=0.5, label='confidence 0.9', tick_label=hands)
plt.bar(x + 0.35, count2, width=0.4, label='confidence 0.8')

plt.legend()
plt.show()
