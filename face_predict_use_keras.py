#-*- coding: utf-8 -*-


import cv2
import sys
import gc
from face_train_use_keras import Model
import os
import path
import pymysql
import time
import pandas as pd
'''

'''


path = "E:\\python\\face_5.17\\data"
l = []
l = os.listdir(path)

model = Model()
model.load_model(file_path = './model/face_detect607.h5')    
       
#框住人脸的矩形边框颜色       
color = (0, 255, 0)
color2 = (255, 0, 0)
color3 = (0, 0, 255)

#捕获指定摄像头的实时视频流
cap = cv2.VideoCapture(0)
#"rtsp://admin:Chen1qaz@192.168.1.108/cam/realmonitor?channel=1&subtype=0"
#人脸识别分类器本地存储路径
cascade_path = "E:/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"    


#conn = pymysql.connect(host='192.168.1.102', port=3306, user='root', passwd='123456',db="dect",charset='utf8')
#cursor = conn.cursor()
#循环检测识别人脸
face_ID = []
name_ID = []
time_ID = []
while True:
    _, frame = cap.read()   #读取一帧视频

#    db = pymysql.connect("192.168.1.102","root","123456","dect",charsetxxxxzxxz = 'utf8')
#    cursor = db.cursor()
    #图像灰化，降低计算复杂度
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #使用人脸识别分类器，读入分类器
    cascade = cv2.CascadeClassifier(cascade_path)                

    #利用分类器识别出哪个区域为人脸
    faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
    if len(faceRects) > 0:                 
        for faceRect in faceRects: 
            x, y, w, h = faceRect
            
            #截取脸部图像提交给模型识别这是谁
            image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            faceID = model.face_predict(image)   
            
            face_ID.append(faceID)
            name_ID.append(l[faceID])
            time_ID.append(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))  
            
            if faceID == 0:                                                        
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                
                #文字提示是谁
                cv2.putText(frame,l[faceID], 
                            (x + 30, y + 30),                      #坐标
                            cv2.FONT_HERSHEY_SIMPLEX,              #字体编码结构光，三班结构光
                            1,                                     #字号
                            (255,0,255),                           #颜色
                            2)                                     #字的线宽
                
            elif faceID ==1:
            #else: 
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color2, thickness = 2)
                
                #文字提示是谁
                cv2.putText(frame,l[faceID], 
                            (x + 30, y + 30),                      #坐标
                            cv2.FONT_HERSHEY_SIMPLEX,              #字体
                            1,                                     #字号
                            (255,0,255),                           #颜色
                            2)                                     #字的线宽
                
            elif faceID ==2:
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color3, thickness = 2)
                
                #文字提示是谁
                cv2.putText(frame,l[faceID], 
                            (x + 30, y + 30),                      #坐标
                            cv2.FONT_HERSHEY_SIMPLEX,              #字体
                            1,                                     #字号
                            (255,0,255),                           #颜色
                            2)                                     #字的线宽      

            elif faceID ==3:
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color3, thickness = 2)
                
                #文字提示是谁
                cv2.putText(frame,l[faceID], 
                            (x + 30, y + 30),                      #坐标
                            cv2.FONT_HERSHEY_SIMPLEX,              #字体
                            1,                                     #字号
                            (255,0,255),                           #颜色
                            2)                                     #字的线宽
        
        
    dataframe = pd.DataFrame({'check_time':time_ID, 'person_name':name_ID, 'faceID':face_ID})
            
            
            
            
            
#            sql_num ="delete from  person_attribute where person_id ="+str(faceID)
#            cursor.execute(sql_num)
#            conn.commit()
#            sql_name ="insert into  person_attribute(person_id, Expression)value("+str(faceID)+","+str(l[faceID])+")"#,"+str( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+")"
#            cursor.execute(sql_name)
#            conn.commit()
    cv2.imshow("show", frame)
    
    #等待10毫秒看是否有按键输入
    k = cv2.waitKey(10)
    #如果输入q则退出循环
    if k & 0xFF == ord('q'):
        break
    
dataframe.to_csv("test.csv",index=False,sep=',')
#释放摄像头并销毁所有窗口
cap.release()
cv2.destroyAllWindows()