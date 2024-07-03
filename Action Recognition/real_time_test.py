
# _______________________________________________________________________
# 1.导入存储的模型
from tensorflow import keras

model_LSTM = keras.models.load_model('recognize_model_LSTM.h5')

# _______________________________________________________________________
# 2.获取实时手动作
import cv2
import mediapipe as mp
import numpy as np
import time
import os

# 调用摄像头
cap = cv2.VideoCapture('output1671017502.1621168.avi')

# 每个手势数据的帧长度
sequences_len = 20
# 手掌检测最小置信度，当检测得分小于这个值时，认为不再存在手（mediapipe参数）
min_det_con = 0.5
# 手追踪最小置信度，当检测得分小于这个值时，重新进行手掌检测（mediapipe参数）
min_track_con = 0.5
path = 'gesture_Data'

# 调用手势识别
mpHands = mp.solutions.hands
# 设置手势检测的参数parameter，有四个参数
# ctrl+ 鼠标左键 可以进入方法的源码，
hands = mpHands.Hands(max_num_hands=1,
                      min_detection_confidence=min_det_con, min_tracking_confidence=min_track_con)
mpDraw = mp.solutions.drawing_utils


actions = os.listdir(path)
# print(actions)  [['pinch', 'press', 'insert', 'screw', 'prod']]
label_map = {label: num for num, label in enumerate(actions)}
print(actions[0])


# 特征点序列集
gesture_keyPoints = []
# 手势判定结果记录
gesture_sentence = []
# 置信度
detection_confidence = float(0.8)
print("手势识别置信度为：{}".format(detection_confidence))

# 每个手势数据的帧长度
sequences_length = sequences_len

if __name__ == "__main__":
    with mpHands.Hands(max_num_hands=1,
                       min_detection_confidence=min_det_con, min_tracking_confidence=min_track_con) as hands:
        while True:
            # 掉取帧画面
            ret, frame = cap.read()
            # make detections
            # 手势的识别图格式是RGB
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)  # 换算后cx，cy含义为在当前帧图像上的像素坐标
                        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    hands_pos = np.array(hands_pos).flatten()
                    mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                    # 绘制hand_landmarks,并绘制关节点连接
            else:
                hands_pos = np.zeros(21 * 3)

            # 特征点记录
            keyPoints = hands_pos
            # 收集30组
            # print("每帧中手掌特征点的shape为：{}".format(keyPoints.shape))
            gesture_keyPoints.insert(0, keyPoints)
            # gesture_keyPoints.append(keyPoints)
            gesture_seq = gesture_keyPoints[:sequences_len]
            current_conf = float(0.0)
            # 当数据帧够数据长度时，就可以进行预测
            if len(gesture_seq) == sequences_len:
                # np.expand_dims()
                # 是由于predict的输入是x_test,这个数据会比gesture_seq多一个维度
                result = model_LSTM.predict(np.expand_dims(gesture_seq, axis=0))[0]
                print(result)
                print(result[np.argmax(result)])
                print(actions[np.argmax(result)])
                # print(type(result[np.argmax(result)]))
                current_conf = result[np.argmax(result)]
            # 当结果中最大值大于检测置信度时：
            if current_conf > detection_confidence:
                # 如果当前有手势判定结果，则和手势判定结果进行比对，不同时则出现了新的手势
                if len(gesture_sentence) > 0:
                    if actions[np.argmax(result)] != gesture_sentence[-1]:
                        gesture_sentence.append(actions[np.argmax(result)])
                # 如果当前没有手势判定结果，则存入一个判定结果
                else:
                    gesture_sentence.append(actions[np.argmax(result)])
            # 只显示最新5个检测结果
            if len(gesture_sentence) > 5:
                gesture_sentence = gesture_sentence[-5:]

            cv2.rectangle(frame, (0, 0), (640, 40), (180, 3, 77), thickness=-1)
            cv2.putText(frame, "  ".join(gesture_sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Opencv Feed", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
