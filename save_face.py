#-*- coding: utf-8 -*-

import cv2
import sys
import os
import time
import datetime
from PIL import Image

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录,创建目录操作函数
        os.makedirs(path) 
 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)                
    
    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("E:\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt2.xml")
    
    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    
    num = 0    
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:            
            break                
    
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像            
        
        #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          #大于0则检测到人脸                                   
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect                        
                
                #将当前帧保存为图片
                img_name = '%s/%d.jpg'%(path_name, num)                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)                                
                                
                num += 1                
                if num > (catch_pic_num):   #如果超过指定最大保存数量退出循环
                    break
                
                #画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                
                #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        
        #超过指定最大保存数量结束程序
        if num > (catch_pic_num): break                
                       
        #显示图像
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows() 
    
    
def acc_save():
    name = str(input("请输入对象姓名 (拼音/英文)"))
    # 定义要创建的目录 
    path_name = "E:/python/face_dataset/%s" %name
    # 调用函数
    mkdir(path_name)
    
    window_name = "image"
    #camera_idx = "rtsp://admin:Chen1qaz@192.168.1.108/cam/realmonitor?channel=1&subtype=0"
    camera_idx = 0
    #"rtsp://admin:Chen1qaz@192.168.1.108/cam/realmonitor?channel=1&subtype=0"
    
    #捕获图像数目
    catch_pic_num = int(input("请输入需要捕获的图像数"))
    CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name)
    
    
    if len(sys.argv) != 4:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("截取人脸", int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

def range_save():
#    hour = int(input("小时范围"))
#    minit = int(input("分钟范围"))
#    sec = int(input("秒范围"))
#    ran = hour + minit + sec
    catch_pic_num = int(input("请输入需要捕获的图像数"))
    

    
    path = 'E:/python/face_dataset' 
    mkdir(path)
    
    window_name = "image"
    camera_idx = 0    
    
    CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path)
    
    
#    ti = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
#    filename = path+ti[-8:]+'.jpg'
    
    if len(sys.argv) != 4:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("截取人脸", int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
        

i = int(input("1、进行精确采集\n2、进行时间范围采集\n3、结束采集\n"))

if i == 1:
    acc_save()
#elif i == 2:
#    range_save()
else:
    print()

