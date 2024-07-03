# 1.下载并导入相关工具包
# import and install Dependencies
#  tensorflow、 mediapipe、 sklearn、 matplotlib、 opencv-python
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
"""
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# ——————————————————————————————————————————————————————————————
# parameter
# 手掌检测最小置信度，当检测得分小于这个值时，认为不再存在手（mediapipe参数）
min_det_con = 0.5
# 手追踪最小置信度，当检测得分小于这个值时，重新进行手掌检测（mediapipe参数）
min_track_con = 0.5
# 文件保存路径,导出数据的路径，numpy数据
dataPath = 'MP_Data'
# 装配动作,动与其他计算机视觉所采用的关键区别在于，检测使用的是一系列数据而不是一帧数据
dynamic_gesture = ['hello', 'thanks', 'i love you']
# 每个手势的训练数据个数
numbers_each_data = 30
# 每个手势数据的帧长度
sequences_len = 30
# ______________________________________________________________
cap = cv2.VideoCapture(0)
# 接受全部种类的识别，包括手势、姿势和面部等等
mp_holistic = mp.solutions.holistic  # holistic model
# mp_hands=mp.solutions.hands   # just hand
# 检测后绘图
mp_drawing = mp.solutions.drawing_utils  # drawing utilities


# 设置识别和追踪置信度
# holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion
    image.flags.writeable = False  # Image is no longer writeable
    result = model.process(image)  # make detection
    image.flags.writeable = True  # Image is writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion RGB 2 BGR
    return image, result


# def draw_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # draw face
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # draw pose
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # draw left hand
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # draw right hand


# 改变绘制时landmark的线和点的风格
#
#
# 这个也需要改进一下
def draw_style_landmarks(image, results):
    # draw face
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
    #                           mp_drawing.DrawingSpec(color=(80, 110, 100), thickness=1, circle_radius=1),
    #                           mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # draw pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    # draw left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # draw right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 122, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 144, 250), thickness=2, circle_radius=2))


#
##
##
##
##
## 这个需要改进一下
def record_results(results):
    # pos = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark].flatten()
    #                if results.pose_landmarks else np.zeros(132))
    # lh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark].flatten()
    #               if results.pose_landmarks else np.zeros(21 * 3))
    # rh = np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark].flatten()
    #               if results.pose_landmarks else np.zeros(21 * 3))
    # return np.concatenate([pos, lh, rh])
    # 记录pose的值为集合
    pos = []
    if results.pose_landmarks:
        for res in results.pose_landmarks.landmark:
            test = np.array([res.x, res.y, res.z]).flatten()
            pos.append(test)
        pos = np.array(pos).flatten()
        # print(pos.shape)
    else:
        pos = np.zeros(99)
    # 记录左手的值为集合
    lh = []
    if results.left_hand_landmarks:
        for res in results.left_hand_landmarks.landmark:
            test = np.array([res.x, res.y, res.z])
            lh.append(test)
        lh = np.array(lh).flatten()
        # print(lh.shape)
    else:
        lh = np.zeros(21 * 3)
    # 记录右手的值为集合
    rh = []
    if results.right_hand_landmarks:
        for res in results.right_hand_landmarks.landmark:
            test = np.array([res.x, res.y, res.z])
            rh.append(test)
        rh = np.array(rh).flatten()
        # print(rh.shape)
    else:
        rh = np.zeros(21 * 3)
    return np.concatenate([pos, lh, rh])  # 返回三者的集合大小为99+63+63=225


def creat_Documents(data_path, gestures, nom_each_data):
    # 4.数据的采集
    # path for export data, numpy data
    DATA_PATH_ = os.path.join(data_path)
    # action detection
    # a key difference between action detection and other computer vision takes is that a sequence of
    # data rather than a single frame is used for detection
    actions_ = np.array(gestures)
    # thirty videos worth for data
    nom_sequences_ = nom_each_data
    # Data
    # just to recap,were going to collect 30 videos per action (hello, thanks, i love you)
    # then each one of those video sequences are going to contain 30 frames of data ,
    # each frame will contain 225 landmarks values
    for action in actions_:
        for seq in range(nom_sequences_):
            try:
                os.makedirs(os.path.join(DATA_PATH_, action, str(seq)))
            except:
                pass


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# outputPath = 'result' + str(time.time()) + '.avi'
# out = cv2.VideoWriter(outputPath, fourcc, 30, (640, 480))

if __name__ == '__main__':

    with mp_holistic.Holistic(min_detection_confidence=min_det_con, min_tracking_confidence=min_track_con) as holistic:
        #
        # 创建文件夹
        creat_Documents(data_path=dataPath, gestures=dynamic_gesture, nom_each_data=numbers_each_data)
        # 实例化
        DATA_PATH = os.path.join(dataPath)
        actions = np.array(dynamic_gesture)
        # 每个手势的训练数据个数
        nom_sequences = numbers_each_data
        # 每个手势数据的帧长度
        sequences_length = sequences_len
        # 将数据写入到文件夹中
        for actions in actions:
            # loop through sequences aka video
            for sequence in range(nom_sequences):
                # loop through video length aka sequence length
                for frame in range(sequences_length):
                    ret, img = cap.read()

                    image, results = mediapipe_detection(img, holistic)  # 返回值为全部的检测结果
                    # print(results.face_landmarks.landmark)

                    draw_style_landmarks(image, results)  # draw landmarks
                    record = record_results(results)
                    print(record.shape)
                    # new apply wait logic
                    if frame == 0:
                        cv2.putText(image, 'START COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames {} for Video Numeber{}'.format(actions, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting frames {} for Video Numeber{}'.format(actions, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    # new export key points
                    npy_path = os.path.join(DATA_PATH, actions, str(sequence), str(frame))
                    np.save(npy_path, record)

                    cv2.imshow('OpenCV Feed', image)
                    # out.write(image)

                    # break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()
