import cv2
import mediapipe as mp
import time
import os
# mediapipe google开发
import numpy as np

cap = cv2.VideoCapture(0)
# 手掌检测最小置信度，当检测得分小于这个值时，认为不再存在手（mediapipe参数）
min_det_con = 0.5
# 手追踪最小置信度，当检测得分小于这个值时，重新进行手掌检测（mediapipe参数）
min_track_con = 0.5
# 文件保存路径,导出数据的路径，numpy数据
dataPath = 'gesture_Data2222222'
# 装配动作,动与其他计算机视觉所采用的关键区别在于，检测使用的是一系列数据而不是一帧数据
dynamic_gesture = ['prod']
 #'pinch' , 'press', 'insert1', 'screw', 'prod'
# 每个手势的训练数据个数
numbers_each_data = 300
# 每个手势数据的帧长度
sequences_len = 60

# 调用手势识别
mpHands = mp.solutions.hands
# 设置手势检测的参数parameter，有四个参数
# ctrl+ 鼠标左键 可以进入方法的源码，
hands = mpHands.Hands(max_num_hands=1,
                      min_detection_confidence=min_det_con, min_tracking_confidence=min_track_con)
mpDraw = mp.solutions.drawing_utils

pre_Time = 0
cur_Time = 0


def create_Documents(data_path, gestures, nom_each_data):
    # 4.数据的采集
    DATA_PATH_ = os.path.join(data_path)

    actions_ = np.array(gestures)

    nom_sequences_ = nom_each_data
    # Data

    for action in actions_:
        for seq in range(nom_sequences_):
            try:
                os.makedirs(os.path.join(DATA_PATH_, action, str(seq)))
            except:
                pass


if __name__ == '__main__':
    with mpHands.Hands(max_num_hands=1,
                       min_detection_confidence=min_det_con, min_tracking_confidence=min_track_con) as hands:

        create_Documents(data_path=dataPath, gestures=dynamic_gesture, nom_each_data=numbers_each_data)
        # 实例化
        DATA_PATH = os.path.join(dataPath)
        actions = np.array(dynamic_gesture)
        # 每个手势的训练数据个数
        nom_sequences = numbers_each_data
        # 每个手势数据的帧长度
        sequences_length = sequences_len

        for actions in actions:
            # loop through sequences aka video
            for sequence in range(nom_sequences):
                # loop through video length aka sequence length
                for frame in range(sequences_length):
                    success, img = cap.read()
                    # 手势的识别图格式是RGB
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(imgRGB)
                    # print(results.multi_hand_landmarks)
                    hands_pos = []
                    if results.multi_hand_landmarks:
                        for handLms in results.multi_hand_landmarks:
                            for id, lm in enumerate(handLms.landmark):
                                # print(id, lm)
                                test = np.array([float(lm.x), float(lm.y), float(lm.z)])
                                # print(test)
                                hands_pos.append(test)
                                # #lm为在当前帧图像上的比例坐标
                                h, w, c = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)  # 换算后cx，cy含义为在当前帧图像上的像素坐标
                                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                            hands_pos = np.array(hands_pos).flatten()
                            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                            # 绘制hand_landmarks,并绘制关节点连接
                    else:
                        hands_pos = np.zeros(21 * 3)

                    record = hands_pos
                    print(record)

                    if frame == 0:
                        cv2.putText(img, 'START COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(img, 'Collecting frames {} for Video Numeber{}'.format(actions, sequence),
                                    (15, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.waitKey(3000)
                    else:
                        cv2.putText(img, 'Collecting frames {} for Video Numeber{}'.format(actions, sequence),
                                    (15, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    # new export key points
                    if frame % 3 == 0:
                        npy_path = os.path.join(DATA_PATH, actions, str(sequence), str(frame))
                        np.save(npy_path, record)

                    cur_Time = time.time()
                    fps = 1 / (cur_Time - pre_Time)
                    pre_Time = cur_Time

                    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                                (255, 0, 255), 3)

                    cv2.imshow("Image", img)
                    cv2.waitKey(1)
