#! /home/hyesong/anaconda3/bin python
# USAGE
# python filter_activity_recognition.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel -i 0

# import the necessary packages
import numpy as np
import argparse
import cv2

from matplotlib import pyplot as plt
# multi-step encoder-decoder lstm 
import tensorflow as tf
from tensorflow import keras
from numpy import array
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import pymysql  #insert data

import time
from datetime import datetime    #second to hour

#from keras.backend.tensorflow_backend import set_session

from flask import Flask, render_template, Response

app = Flask(__name__)


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#`config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
session = tf.compat.v1.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras



#About SSD
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--prototxt", required=False,default="MobileNetSSD_deploy.prototxt.txt",
#    help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=False, default="MobileNetSSD_deploy.caffemodel",
#    help="path to Caffe pre-trained model")
#ap.add_argument("-i", "--camindex", default=0, type=int,
#    help="Camera Index for video capturing")      
#ap.add_argument("-c", "--confidence", type=float, default=0.2,
#    help="minimum probability to filter weak detections")
#args = vars(ap.parse_args())
conf = 0.2

#camIndex = args["camindex"]

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
#detect only background and human
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",        
           "sofa", "train", "tvmonitor"]
IGNORE= set(["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"])

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
LABELS = ["punch", "shoot_gun", "person"]
    
x_grid = 20
y_grid = 32
size_grid = x_grid*y_grid
size_sample = 15
size_batch = 20
    

#load the SSD model
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net=cv2.dnn.readNetFromCaffe("/home/hyesong/learning_eyes/Activity_Recognition/MobileNetSSD_deploy.prototxt.txt",                                    "/home/hyesong/learning_eyes/Activity_Recognition/MobileNetSSD_deploy.caffemodel")
    
#load the LSTM model
model = load_model('/home/hyesong/learning_eyes/Activity_Recognition/model50_50_1113.h5') 
    
    
    
###############HERRREEEEEEEEEEEEEEEEEEArgument should take the number(index) of camera
   
# initialize the video 
#cap = cv2.VideoCapture('C:/Users/zoclz/clappingTest2.avi')
#cap = cv2.VideoCapture(0)
#camIndex should come here for camera 0, 1, 2, .... 
#we should set several cameras with the camList and choose through the index
#but we have only one cam for this, so we don't use this part. 
#just for extension
videoFile = '/home/hyesong/learning_eyes/Activity_Recognition/test2.avi'

cap = cv2.VideoCapture(videoFile)
#cap = cv2.VideoCapture(0)


    
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('cam.html')




#camera number maybe as a param here
def recognition():

    number=1;

    # loop over the frames from the video stream
    count_frame = -1  #count of human frame
    label = ""
    # {'center', 'index_frame', 'frame_count', 'rgbs'} <<<< 이 순서 리스트
    boxList = []
    
    #Code for testing with the video    
    #start
    #startTime = time.time()
        
    while cap.isOpened():
        
        #frame number reset *warning, accurcy may decrease*
        if count_frame == 2000 :
            count_frame = 0
            print(":0")
        
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        #frame = vs.read()
        success, frame = cap.read()
       
        if (success != True):
             
            break
        frame = cv2.resize(frame, (600,360), interpolation=cv2.INTER_AREA) ##GPU OUTOFMEMORY
                   
        count_frame += 1      #프레임 인덱스는 0부터 시작               
        
                         
                         
                         
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
               0.007843, (300, 300), 127.5)
    
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
    
        
        duplicated_center_check = [-1,-1]# 프레임 내 중복 방지 체크 *****    
        nlimit = 0 # over, just skip... for the first break point
        current_centers = []    #현재 프레임의 박스 센터 리스트
        
        boxListBefore = boxList.copy()  #boxListForOnly, for 루프 돌릴 용도의 리스트. 이전 프레임 박스 리스트인거.
        
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):  
            
            
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > conf:
                        
                # extract the index of the class label from the
                # `detections`
                idx = int(detections[0, 0, i, 1])
    
                # if the predicted class label is in the set of classes
                # we want to ignore then skip the detection
                if CLASSES[idx] in IGNORE:
                    continue
                     
                        
                # compute the (x, y)-coordinates of the bounding box(bBox) for
                # the object
                bBox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = bBox.astype("int")
     
                    
                if startX < 0 :
                    startX = 0
                if startY < 0 :
                    startY = 0
                if endX > len(frame[0]) :
                    endX = len(frame[0])
                if endY > len(frame) :
                    endY = len(frame)
                            
                humanbox = frame[startY:endY, startX:endX]
                try : 
                    humanbox = cv2.resize(humanbox, (x_grid, y_grid), interpolation = cv2.INTER_NEAREST)
                except Exception as e:
                    print(str(e))
                    continue
                            
                            
                ######nlimit check#####
                nlimit += 1
                if (nlimit > 50) :   #speed up
                    print("too much") #.... second break point will not appear with this break point
                    break
    
    
    
    
    
                ######################################    Box Check, Get Grid Points    ###################################
                    
                ######################### CASE 1 : 이전 프레임에 박스들이 존재하며 그 박스들 중 이어지는 행동의 박스로 판단
                
                if len(boxListBefore) > 0:    #이전 프레임에 박스 후보군이 존재하면
                    
                    box_count = 0
                    
                    for box in boxListBefore : #중앙 점이 비슷해야함              
                        ######################    +- 5 오차 설정 포인트~~~~    ##################################
                        #사람이 휘두르는 속도... 고려해야겠는걸....    
                        
                        box_count += 1
                        if box_count > 50:  #speed up
                            #print("too many before_centers")
                            break
                            
                            
                            
            
                        if ( (((startY+endY)/2 < box.get("center")[0] + 10) 
                              and ((startY+endY)/2 > box.get("center")[0] - 10) ) 
                            and (((startX+endX)/2 < box.get("center")[1] + 10) 
                              and ((startX+endX)/2 > box.get("center")[1] - 10) )):  
    
                            
                            #이전 프레임에 있던 박스와 같은 박스라고 판단 되었을 경우, 
                            #프레임 차이를 계산할 것.
                            #새 세그먼트가 아닐 경우이므로 segment_number는 변화 x
                                    
                                   
                            #프레임 내 중점 중복 체크  (없는것보다 있는게 빠르고... 3했다가 5해봄. 큰 차이는 x)
                            if ( ((duplicated_center_check[0] <= (startY+endY) / 2 + 5) 
                                   and (duplicated_center_check[0] >= (startY+endY) / 2 - 5))
                                      and ((duplicated_center_check[1] <= (startX+endX)/2 + 5)  
                                   and (duplicated_center_check[1] >= (startX+endX)/2 - 5)) ) :
                                continue
                      
        
                            #currentbox = []                           
                            currentbox = [ frame[y_i, x_i, :3].dot([0.299, 0.587, 0.114]) 
                                          for x_i in range(x_grid) for y_i in range(y_grid) ]
    
        
        
                            #####박스 목록에 세팅 업데이트     인덱싱 말고 행으로
                        
                            
                            updateA = [(startY+endY)/2,(startX+endX)/2]       #center
                            updateB = count_frame                        #index_frame
                            updateC = box.get("frame_count") + 1                      #frame_count                   
                            updateD = box.get("rgbs").copy()                          #rgbs
                            updateD.append(currentbox)                          
                            box.update( center = updateA,
                                        index_frame = updateB,
                                        frame_count = updateC,
                                        rgbs = updateD)
    
                            
                            
                            ##################만약 16번째 프레임일 경우(차이 총 15개) 예측##################               
                            if box.get("frame_count") > size_sample : 
                                
                                #predict
                                inputchange = []
                                iappend = inputchange.append   #faster
                                boxGetRgb = box.get("rgbs")                            
                                for j in range(size_sample): 
                                    iappend(list(map(lambda k : boxGetRgb[j+1][k]-boxGetRgb[j][k], range(size_grid)))) 
                                 
                                x_input = np.array([inputchange])                         
                                x_input = x_input.reshape((1, size_sample, size_grid)) #time_step이 여기선 1이니까..? 
                                #print(x_input.shape)                          
                                y_p = model.predict(x_input,batch_size= 1, verbose=0) 
                                    #model batch 사이즈 정하는거 포기했으니 됬겠지.. 배치 그냥... train때도 1로 하자...... 아 batch 줄면 줄어드는데... accuracy 아마...
    
                                y_list = list(y_p[0])  #1d array, list() instead of tolist
                                print(y_list)                            
                                if y_list.index(max(y_list)) != len(LABELS)-1 :   
                                    #print(LABELS[y_list.index(max(y_list))] + " detected")                              
                                    label = "{}: {:.2f}%".format(LABELS[y_list.index(max(y_list))], max(y_list))
                                    cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                                    y = startY - 15 if startY - 15 > 15 else startY + 15
                                    cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2) 
                                    
                                   
                                    #HEEEEEEEEEEEEEEEEEEREEEEEEEEE! detect signal should go*****************
                                    cv2.imwrite("/var/www/learningeyes/img/" + str(number) + ".png", frame)
                                    number+=1

                                    #db setting
    
                                    con = pymysql.connect(host="localhost", user="root", password="", db='LearningEyes', charset='utf8')
                                    cur = con.cursor()
                                    detectDate = datetime.now().strftime("%Y-%m-%d")
                                    detectTime = datetime.now().strftime("%H:%M:%S")
    
                                    sql="insert into detect_log(logDate, LogTime, screenNumber, screenPlace, detectAction) values (%s, %s, %s, %s, %s)"
                                    cur.execute(sql, (detectDate, detectTime, '21', 'Main Hall',
                                                  LABELS[y_list.index(max(y_list))]))
                                    con.commit()
                                
                            duplicated_center_check[0] = (startY+endY)/2
                            duplicated_center_check[1] = (startX+endX)/2
                         
                                    
                        ######################### CASE 2 : 이전 프레임에 박스들이 존재하지만 새 행동(df row)이 시작됨
                     
                        else : #이전 프레임에 존재하지 않는 박스가 탐지된 경우, current에 일단 저장                      
                                                      
                            currentbox = [ frame[y_i, x_i, :3].dot([0.299, 0.587, 0.114]) 
                                          for x_i in range(x_grid) for y_i in range(y_grid) ]
                                         
                            boxList.append( {'center' : [(startY+endY)/2,(startX+endX)/2],
                                      'index_frame' : count_frame,
                                      'frame_count' : 1,
                                      'rgbs' : [currentbox]} )                 
                                    
                        ######################### CASE 3 : 이전 프레임에 박스들이 존재 하지 않았고, 새 행동(df row)이 시작됨
                                                       
                else :### 처음으로 박스가 탐지된 경우, current에 일단 저장
                         
                            
                    currentbox = [ frame[y_i, x_i, :3].dot([0.299, 0.587, 0.114]) 
                                  for x_i in range(x_grid) for y_i in range(y_grid) ] 
                  
                     
                    boxList.append( {'center' : [(startY+endY)/2,(startX+endX)/2],
                              'index_frame' : count_frame,
                              'frame_count' : 1,
                              'rgbs' : [currentbox]} )         
    
        #current frame을 최신으로 가지고 있는 녀석만 남기고 데이터프레임 리셋
        
        boxList = list(filter(lambda x : x['index_frame'] == count_frame, boxList)) 
                                                       
                 
        # show the output frame ... too slow to show
        #cv2.imshow("Frame", frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') #concat frame one by one and show result
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        #if cv2.getWindowProperty('Frame', 0) < 0 :
           # break            
          
    cap.release()
    
    #testing for Video
    #endTime = time.time() - startTime
    #hourtime = str(datetime.timedelta(seconds = endTime))                   
    #print("====== " + hourtime + " ======")
     
                       
    
    # do a bit of cleanup
    #cv2.destroyAllWindows()



@app.route('/video_feed')
def video_feed():
    """Video streaming route, put this in the src attribute of an img tag."""
    return Response(recognition(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


        
if __name__ == '__main__':
    app.run(host='0.0.0.0')
        
