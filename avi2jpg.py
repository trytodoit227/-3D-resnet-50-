import os
import glob
import cv2
home_path = os.getcwd()#返回当前进程的工作目录
jpg_path = os.path.join(home_path,'work','ucf101')#打开图像存放路径
video_path = os.path.join(home_path,'data','UCF-101')#打开视频存放路径
video_class_list = glob.glob(video_path+'/*')#一次性获取一个文件夹内所有文件的地址！并把地址转为字符串形式
#handstandingpushup这个视频要把testlist中的stand改成Stand，特别坑爹
n=0
all_video=0
for item in video_class_list:
    n=n+1
    
    video_list = glob.glob(item+'/*.avi')
    print(item.split('/')[-1],n)
    for avi in video_list:
        print(avi)
        name = avi.split('/')[-1]
        name = name.split('.')[0]
        jpg_folder = os.path.join(jpg_path,name)
        print(jpg_folder)
        if(not os.path.exists(jpg_folder)):
            os.mkdir(jpg_folder)
        cap = cv2.VideoCapture(avi)#打开该路径下的视频
        No = 0
        while(cap.isOpened()):# 判断载入的视频是否可以打开
            ret, frame = cap.read()#获cap.read()方法的两个返回值。其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵
            if(ret is False):
                break
            No = No+1
            cv2.imwrite(os.path.join(jpg_folder,'frame{:06d}.jpg'.format(No)),frame)#保存一个图像。第一个参数是要保存的文件名，第二个参数是要保存的图像
