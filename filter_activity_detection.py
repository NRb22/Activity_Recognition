# USAGE
# python filter_activity_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2

from matplotlib import pyplot as plt
# multi-step encoder-decoder lstm 
import tensorflow as tf
from numpy import array
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

size_grid = 24
x_grid = 4
y_grid = 6
size_sample = 30
size_batch = 10

#load the LSTM model
model = load_model('C:/Users/zoclz/models/2finalmodel2.h5') 

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

# initialize the video
cap = cv2.VideoCapture('C:/Users/zoclz/clappingTest2.avi')


# loop over the frames from the video stream
count_frame = 0  #count of human frame
beforeframe = []

label = ""
    
while cap.isOpened():
  
	score = 0.0  #for evaluation, reset as well    
    
# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	#frame = vs.read()
	ret, frame = cap.read()
	if (ret != True):
		break
	frame = cv2.resize(frame, None, fx =0.3, fy=0.3, interpolation=cv2.INTER_AREA)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	currentframe = []

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
            
###########################
			count_frame += 1

			#store grid points' pixel difference
			#human part will divded 10 x 20 
			x_div = (endX - startX) / x_grid
			y_div = (endY - startY) / y_grid
            
			index = -1
			#int(startX + float(x_i*float((endX - startX)/4)))
			for x_i in range(0, x_grid):
				for y_i in range(0, y_grid): 
					index +=1                    
					x_c = startX + int(x_i*x_div)
					y_c = startY + int(y_i*y_div)
					gray = frame[y_c, x_c, :3].dot([0.299, 0.587, 0.114])
					currentframe.append(gray)
                    
			if count_frame <= size_sample : #from count_frame == 21, beforeframe has 20 inputs
				beforeframe.append(currentframe)
				continue                
			else:                
				#predict                
				startframe = count_frame - size_sample
				t_step = size_sample - 1
				inputframe = []
				for j in range(startframe, startframe + t_step):
					inputframe.append([beforeframe[j][k] - beforeframe[j-1][k] for k in range(size_grid)])

				inputs = []                    
				for j in range(30) : inputs.append(inputframe)                    
				x_input = np.array(inputs)                
				x_input = x_input.reshape((30, t_step, size_grid))                            
				yhat = model.predict(x_input, verbose=0)                
				predictframe = [beforeframe[t_step][k] + yhat[k] for k in range(size_grid)]
                
                #after the prediction
				beforeframe.append(currentframe)                
                
				####evaluation, round(255 /100*5) = 13, scroe += 100 / size_grid 
				for j in range(size_grid):
					if (currentframe[j]>= predictframe[j] - 13) & (currentframe[j] <= predictframe[j] +13) :
						score += 100/size_grid

                        
				# draw the prediction on the frame                        
			if score > 80:
				print('--------clapping detected--------')
				label = "{}: {:.2f}%".format('clapping person',
					score)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)       
                
			#when there's no prediction                
			else : 
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100) 
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
             
	# show the output frame ... too slow to show
	cv2.imshow("Frame", frame)

    
    
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
        
cap.release()
                   

# do a bit of cleanup
cv2.destroyAllWindows()