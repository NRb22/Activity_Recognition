# USAGE
# python data_creation.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
import numpy as np    # for mathematical operations

import glob #to get video files

import argparse  #train
import time
import datetime    #second to hour


#About SSD
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
#detect only background and human
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
IGNORE = set(["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"])
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

LABELS = ["clap", "jump", "punch", "shoot_gun"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#Make an array of frames
count_video = []
for l_index in range(len(LABELS)):
    count_video.append(0)
total_frames = []
#for the grid, human body ratio 1:1.618 ... 5:8
x_grid = 20
y_grid = 32
size_grid = x_grid*y_grid



#start
startTime = time.time()


mere_count = 0

#######################################    Load Videos, Capture Frames    ############################################

for l_index in range(len(LABELS)):
    frames = []
    for videoFile in glob.glob('dataset/'+LABELS[l_index]+"/*.avi"):
        print('v%d'%mere_count + videoFile)
        mere_count += 1
        
        cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
        x=1
        framesTemp = []
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            framesTemp.append(frame)
        frames.append(framesTemp)   #to erase first []
        cap.release()
        count_video[l_index] += 1
    total_frames.append(frames) 
 
print("Sample Frames Capture Finished\n")

####################################    Collect Activity Grid Points Data    ##########################################

#dataset file creation
f = open("createdData50_50_fin.txt", "w+")

for l_index in range(len(LABELS)):
    
    ##### 각 행동 Segments 뽑아서 저장. (video number, segment number, frame number : ... ... ... ...)  #####
    
    for v_index in range(0, count_video[l_index]):
  
        segment_number = 0    #일종의, 이전 프레임의 저 박스와 같은 박스가 움직인거냐~ 체크하기 위해 그 한 박스 행동 세그먼트를 이렇게 표시.
        segments = []

        count_frame = 0       #프레임 넘버
        before_centers = []  #이전 프레임의 박스 센터 리스트
        before_boxes = []     #이전 프레임의 박스 센터 리스트
        
        
        
    
    

        for frame in total_frames[l_index][v_index]:
            
            nlimit = 0 # over, just skip... for the first break point
    
    
    
            current_centers = []    #현재 프레임의 박스 센터 리스트
            current_boxes = []      #현재 프레임의 박스들 리스트
            
            duplicated_center_check = [-1,-1]# 프레임 내 중복 방지 체크 *****
            
        
            #For performance, resize
            #frame = cv2.resize(frame, None, fx =0.2, fy=0.2, interpolation=cv2.INTER_AREA)
            #frame = imutils.resize(frame, width=400)

            ##############################    SSD, Human Box Detection    ##############################################
            
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward() 

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > args["confidence"]:
                    # extract the index of the class label from the
                    # `detections`
                    idx = int(detections[0, 0, i, 1])

                    # if the predicted class label is in the set of classes
                    # we want to ignore then skip the detection
                    if CLASSES[idx] in IGNORE:
                        continue

                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                
                
                    if startX < 0 :
                        startX = 0
                    if startY < 0 :
                        startY = 0
                    if endX > len(frame[0]) :
                        endX = len(frame[0])
                    if endY > len(frame) :
                        endY = len(frame)
                        
                        
                    #this data_creation, there's no point (x, y) so just y, x for lists..
                    humanbox = frame[startY:endY, startX:endX]
                    try : 
                        humanbox = cv2.resize(humanbox, (x_grid, y_grid), interpolation = cv2.INTER_NEAREST)
                    except Exception as e:
                        print(str(e))
                        continue
                        
                        
                    ######nlimit check#####
                    nlimit += 1
                    if (nlimit > 50) :
                        print("too much") #.... second break point will not appear with this break point
                        break
                
                
                    ######################################    Box Check, Get Grid Points    ###################################
                
                    ######################### CASE 1 : 이전 프레임에 박스들이 존재하며 그 박스들 중 이어지는 행동의 박스로 판단
                    #Center와 Box는 동시 저장, index 공유
                
                    if len(before_centers) > 0:    #박스 탐지가 처음이 아니면?
                        box_count = 0

                        for c_index in range(len(before_centers)):   #기존에 이어지던 행동? 중앙 점이 비슷하고 프레임 넘버 차이가 1이어야 함.
                            ######################    +- 5 오차 설정 포인트~~~~    ##################################
                            
                            box_count+= 1
                            if box_count > 50:
                                #print("too many before_centers")
                                break
                            
                            
                            
                            if ( ( ((startY+endY)/2 < before_centers[c_index][0] + 10) and ((startY+endY)/2 > before_centers[c_index][0] - 10) ) and
                                 ( ((startX+endX)/2 < before_centers[c_index][1] + 10) and ((startX+endX)/2 > before_centers[c_index][1] - 10) )  
                                and (count_frame - before_centers[c_index][3] == 1)) : 
                                #print("!")
                                #이전 프레임에 있던 박스와 같은 박스라고 판단 되었을 경우, 
                                #프레임 차이를 계산할 것.
                                #새 세그먼트가 아닐 경우이므로 segment_number는 변화 x
                                
                                

                                #프레임 내 중점 중복 체크  (없는것보다 있는게 빠르고... 3했다가 5해봄. 큰 차이는 x)
                                if ( ((duplicated_center_check[0] <= (startY+endY) / 2 + 5) 
                                   and (duplicated_center_check[0] >= (startY+endY) / 2 - 5))
                                      and ((duplicated_center_check[1] <= (startX+endX)/2 + 5)  
                                   and (duplicated_center_check[1] >= (startX+endX)/2 - 5)) ) :
                                    continue
                  
                               
                                current_centers.append([(startY+endY)/2,(startX+endX)/2, before_centers[c_index][2], count_frame])
                                
                                currentbox = [ frame[y_i, x_i, :3].dot([0.299, 0.587, 0.114]) 
                                              for x_i in range(x_grid) for y_i in range(y_grid) ]
                                        
                                current_boxes.append(currentbox)
                                
                                #### 여기서 차이 계산 ####
                                ######### 그리고 유의미한 행동 데이터 저장 ########
                                #label, video number , segment number, frame number : grid points...
                                f.write(LABELS[l_index])
                                f.write(" %d %d %d :" % (v_index+1, before_centers[c_index][2], count_frame))
                                
                                for k in range(0, size_grid):  
                                    f.write(" %d" % int(currentbox[k] - before_boxes[c_index][k]))
                                
                                f.write("\n")
                                ######### 유의미한 행동 데이터 저장 ########
                                
                                duplicated_center_check[0] = (startY+endY)/2
                                duplicated_center_check[1] = (startX+endX)/2
                         
                                
                     ######################### CASE 2 : 이전 프레임에 박스들이 존재하지만 그 박스 중에 없는 새 segment의 행동  
                                                   
                            else : #이전 프레임에 존재하지 않는 박스가 탐지된 경우, current에 일단 저장
                                #print("-")
                                current_centers.append([(startY+endY)/2,(startX+endX)/2, segment_number, count_frame])
                                segment_number += 1 # 새 세그먼트임
                        
                                currentbox = [ frame[y_i, x_i, :3].dot([0.299, 0.587, 0.114]) 
                                              for x_i in range(x_grid) for y_i in range(y_grid) ]
                                        
                                current_boxes.append(currentbox)
                                
                    ######################### CASE 3 : 이전 프레임에 박스들이 존재 하지 않았고, 새 segment의 행동이 시작됨
                                                   
                    else :### 처음으로 박스가 탐지된 경우, current에 일단 저장
                        #print(".")
                        current_centers.append([(startY+endY)/2,(startX+endX)/2, segment_number, count_frame])
                        segment_number += 1 # 새 세그먼트임
                        
                        
                        currentbox = [ frame[y_i, x_i, :3].dot([0.299, 0.587, 0.114]) 
                                              for x_i in range(x_grid) for y_i in range(y_grid) ]
                        
                        current_boxes.append(currentbox)
                        
                        
            #end of per detection                           
                  
            count_frame +=1
            before_centers = np.copy(current_centers)
            before_boxes = np.copy(current_boxes)
                              
        #end of per frame
        print("%d th video finished--" % v_index)        
        
    #finishing one activity data    
    
print("\n-----video difference array finished----\n") #clap 130 videos checked 0~ 129   
    
f.close()                      

endTime = time.time() - startTime
hourtime = str(datetime.timedelta(seconds = endTime))                   
print("====== " + hourtime + " ======")
 
