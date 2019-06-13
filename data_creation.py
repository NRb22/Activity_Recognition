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
IGNORE = set(["aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"])
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#Make an array of frames
count_video = 0
frames = []
#for the grid
x_grid = 2
y_grid = 3
size_grid = 6

for videoFile in glob.glob('D:/Nuri/2019School/DatabaseSystem/videos/selected/clap/clap/*.avi'):
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    x=1
    framesTemp = []
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        framesTemp.append(frame)
    frames.append(framesTemp)
    cap.release()
    count_video += 1
    
print("Sample Frames Capture Finished\n")

human_data = []  #total videos' frames' grid points difference


for v_index in range(0, count_video):
	video_data = []  #total frames' grid points difference
	beforeframe = []
	count_frame = 0    

	for frame in frames[v_index]:
        
		frame = cv2.resize(frame, None, fx =0.2, fy=0.2, interpolation=cv2.INTER_AREA)
		#frame = imutils.resize(frame, width=400)

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
				croped = np.copy(frame[startY:endY, startX:endX])
                
            
				#store grid points' pixel difference
				#human part will divded 4 x 6 
				currentframe = []
				x_div = (endX - startX) / x_grid
				y_div = (endY - startY) / y_grid
            
				#int(startX + float(x_i*float((endX - startX)/x_grid)))
				for x_i in range(0, x_grid):
					for y_i in range(0, y_grid): 
						x_c = startX + int(x_i*x_div)
						y_c = startY + int(y_i*y_div)
						gray = frame[y_c, x_c, :3].dot([0.299, 0.587, 0.114])
						currentframe.append(gray)                
                               
				if count_frame == 0:
					beforeframe = np.copy(currentframe)
					count_frame +=1
					continue
            
				framechange = []
				for k in range(0, size_grid):
					framechange.append(int(currentframe[k] - beforeframe[k]))
                
				video_data.append(framechange)
				count_frame +=1
				beforeframe = np.copy(currentframe)

			#end of per detection             
		#end of per frame
	print("%d th video finished--" % v_index)
	#finally one video's frame's meaningful pixel difference will be stored
	human_data.append(video_data)
    
print("\n-----video difference array finished----\n")    
    
    #dataset file creation
f = open("clap3.txt", "w+")

for v_index in range(0, count_video):  #video number
	count_frame = len(human_data[v_index])
	for f_index in range(0, count_frame):  #frame number
		f.write("%d %d :" % (v_index + 1, f_index + 1))
		for k in range(0, size_grid):
			f.write(" %d" % human_data[v_index][f_index][k])
		f.write("\n")
#1 1 : 0 0 20 0 30 0 5 ...
f.close()                      