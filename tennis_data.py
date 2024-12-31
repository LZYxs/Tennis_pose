import cv2
import mediapipe as mp
import numpy as np
import os
import csv #调用数据保存文件
import pandas as pd #用于数据输出
def calculate_angle(a, b, c):
    '''
    计算角度
    :param a:
    :param b:
    :param c:
    :return:
    '''

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_dist(a, b):
    '''
    计算欧式距离
    :param a:
    :param b:
    :return:
    '''
    #该处考虑改为绝对距离
    a = np.array(a)
    b = np.array(b)
    dist = np.linalg.norm(a - b)#默认参数(矩阵2范数，不保留矩阵二维特性)
    return dist

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    #data = pd.read_csv('test.csv')
    #print(data)
    #pic = cv2.imread("1.png")
    test=[]
    counter = 0
    stage = None
    dir_path = "D:\\tennis_image\\serve"
    files = os.listdir(dir_path)
    for file in files:
        #print(file)
        pic = cv2.imread(dir_path + "\\" + file)
        # 分辨率
        size = pic.shape
        pic_width = size[1]
        pic_height = size[0]
        with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.8) as pose:
            # ret, frame = cap.read()
            frame = pic
            # 转换下颜色空间
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 这里设置为不可写
            image.flags.writeable = False
            # 检测
            results = pose.process(image)
            # 这里设置为可写，颜色也转换回去
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # 提取关键点
            try:
                landmarks = results.pose_landmarks.landmark

                # 获取相应关键点的坐标
                # 左肩11
                lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                # 左手肘13
                lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                # 左手腕15
                lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                # 左胯23
                lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                # 左膝盖25
                lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                # 左脚踝27
                lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                # 右肩膀12
                rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                # 右手肘14
                relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                # 右手腕16
                rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                # 右胯24
                rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                # 右膝盖26
                rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                # 右脚踝28
                rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # 计算角度和距离
                langle = calculate_angle(lshoulder, lelbow, lwrist)  # 左胳膊角度
                rangle = calculate_angle(rshoulder, relbow, rwrist)  # 右胳膊角度
                lsangle = calculate_angle(lhip, lshoulder, lelbow)  # 左臂离身体角度
                rsangle = calculate_angle(rhip, rshoulder, relbow)  # 右臂离身体角度
                ankdist = calculate_dist(lankle, rankle)  # 左右脚踝距离
                rwdist = calculate_dist(rhip, rwrist)  # 右胯到右手腕距离
                lwdist = calculate_dist(lhip, lwrist)  # 左胯到左手腕距离
                lhangle = calculate_angle(lshoulder, lhip, lknee)  # 左肩膀-胯-膝盖角度
                rhangle = calculate_angle(rshoulder, rhip, rknee)  # 右肩膀-胯-膝盖角度
                lkangle = calculate_angle(lankle, lknee, lhip)  # 左腿角度
                rkangle = calculate_angle(rankle, rknee, rhip)  # 右腿角度

                #print(rangle, " ", langle)
                test.append(
                    [langle, rangle, lsangle, rsangle, ankdist, rwdist, lwdist, lhangle, rhangle, lkangle, rkangle,2])
                #print(test)

                #test.to_csv('test.csv')  # 数据存入csv,存储位置及文件名称
            except:
                pass

            cv2.rectangle(image, (0, 0), (100, 30), (235, 30, 3), -1)
            cv2.putText(image, 'STAGE', (12, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # 画骨骼关键点
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # 显示结果帧
            cv2.imshow('mediapipe demo', image)
            #cv2.waitKey()
            # 保存结果帧
            cv2.imwrite(dir_path + "\\" +"res"+ file, image)
            # 资源释放
            # cv2.destroyAllWindows()
    column = ['langle', 'rangle', 'lsangle', 'rsangle', 'ankdist', 'rwdist', 'lwdist', 'lhangle', 'rhangle', 'lkangle',
              'rkangle', 'class']  # 列表头名称,11个特征1个分类
    data = pd.DataFrame(columns=column, data=test)  # 将数据放进表格
    data.to_csv('data.csv', mode='a', index=False)