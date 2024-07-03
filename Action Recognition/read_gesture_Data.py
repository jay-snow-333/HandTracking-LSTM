import os
import numpy as np

path = 'gesture_Data'

actions = os.listdir(path)
# print(actions)  [['pinch', 'press', 'insert', 'screw', 'prod']]
label_map = {label: num for num, label in enumerate(actions)}
print(actions[0])

# x和label
x_sequences, labels = [], []
for num, action in enumerate(actions):
    sequences = os.listdir(path + '/' + str(actions[num]))  # 获得下一级目录的各个文件夹
    # print(len(sequences))  # 300
    for sequence in range(len(sequences)):
        sequnece_length = os.listdir(path + '/' + str(actions[num] + '/' + str(sequence)))  # 获得下下一级目录的各个文件夹
        x = []  # x由 20 x .npy  组合成
        if len(sequnece_length) != 20:
            print(len(sequnece_length))
            print("{}/{}/{}".format(path, actions[num], sequence))
        for frame_key in range(len(sequnece_length)):
            res = np.load(os.path.join(path, action, str(sequence), '{}.npy'.format(frame_key * 3)))
            if len(res) != 63:
                print(len(res))
            print(res)
            x.append(res)
        x_sequences.append(x)

        labels.append(label_map[action])