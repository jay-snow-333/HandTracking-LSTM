import cv2
import mediapipe as mp
import time

# mediapipe google开发
cap = cv2.VideoCapture('fsfer4445dfd.mp4')
cap.set(3, 640)
cap.set(4, 480)

# 调用手势识别
mpHands = mp.solutions.hands
# 设置手势检测的参数parameter，有四个参数
# ctrl+ 鼠标左键 可以进入方法的源码，
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pre_Time = 0
cur_Time = 0

# 保存测试结果
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outputPath = 'result' + str(time.time()) + '.avi'
out = cv2.VideoWriter(outputPath, fourcc, 30, (640, 480))

while True:
    success, img = cap.read()
    # 手势的识别图格式是RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            record = {}
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                # #lm为在当前帧图像上的比例坐标
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # 换算后cx，cy含义为在当前帧图像上的像素坐标
                print(id, cx, cy)
                record[id] = (cx, cy)
                if id == 4 or id == 8 or id == 12 or id == 16 or id == 20 or id == 0 or id == 5 or id == 17:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            cv2.line(img, record[0], record[4], (0, 255, 0), thickness=3)
            cv2.line(img, record[0], record[8], (0, 255, 0), thickness=3)
            cv2.line(img, record[0], record[12], (0, 255, 0), thickness=3)
            cv2.line(img, record[0], record[16], (0, 255, 0), thickness=3)
            cv2.line(img, record[0], record[20], (0, 255, 0), thickness=3)

            cv2.line(img, record[4], record[8], (0, 255, 0), thickness=3)
            cv2.line(img, record[8], record[12], (0, 255, 0), thickness=3)
            cv2.line(img, record[12], record[16], (0, 255, 0), thickness=3)
            cv2.line(img, record[16], record[20], (0, 255, 0), thickness=3)

            cv2.line(img, record[0], record[5], (255, 0, 0), thickness=3)
            cv2.line(img, record[0], record[17], (255, 0, 0), thickness=3)
            cv2.line(img, record[5], record[17], (255, 0, 0), thickness=3)

            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # 绘制hand_landmarks,并绘制关节点连接

    cur_Time = time.time()
    fps = 1 / (cur_Time - pre_Time)
    pre_Time = cur_Time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    # cv2.putText(img, seq_str, (3, 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
    # cv2.rectangle(img, (0, 0), (1080, 40), (180, 3, 77), thickness=-1)
    cv2.imshow("Image", img)
    out.write(img)

    cv2.waitKey(1)
