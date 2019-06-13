# USAGE
# python filter_activity_recognition.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

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

size_grid = 6
x_grid = 2
y_grid = 3

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
cap = cv2.VideoCapture('C:/Users/zoclz/clappingTest.avi')
# write on video
fcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('detected.avi', fcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# loop over the frames from the video stream
count = 0  #count of frame
beforeframe = []
for i in range(0,size_grid):
	beforeframe.append([])
label = ""
    
while cap.isOpened():
    
	score = 0.0  #for evaluation    
    
# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	#frame = vs.read()
	ret, frame = cap.read()
	if (ret != True):
		break
	frame = cv2.resize(frame, None, fx =0.3, fy=0.3, interpolation=cv2.INTER_AREA)
	count += 1
    

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
            
###########################

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
                    
					if count <= 5:
						beforeframe[index].append(gray)
					else:
						temp0 = beforeframe[index][1]
						temp1 = beforeframe[index][2]
						temp2 = beforeframe[index][3]
						temp3 = beforeframe[index][4]                        
						beforeframe[index][0] = temp0
						beforeframe[index][1] = temp1
						beforeframe[index][2] = temp2
						beforeframe[index][3] = temp3  
						beforeframe[index][4] = gray                        
                   
						# demonstrate prediction
						for i in range(0,size_grid):
							model = load_model('C:/Users/zoclz/models/6model%d.h5'%i)
							temp0 = beforeframe[i][1] - beforeframe[i][0]
							temp1 = beforeframe[i][2] - beforeframe[i][1]
							temp2 = beforeframe[i][3] - beforeframe[i][2]
							x_input = array([temp0, temp1, temp2])
							x_input = x_input.reshape((1, 3, 1))
							yhat = model.predict(x_input, verbose=0)
							yhat += beforeframe[i][3]        
							print("%d th model finished"%i)
							del model                    
							####evaluation, round(255 /100*5) = 13, scroe += 100 / size_grid 
							if (gray>= yhat - 13) & (gray <= yhat +13) :
								score += 100/size_grid

                        
			# draw the prediction on the frame                        
			if score > 60:
				print('--------clapping detected--------')
				label = "{}: {:.2f}%".format('clapping person',
					score)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)       
				cv2.imwrite('detectedclappingfrom%dthframe.jpg'%count, frame)
                
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
	#cv2.imshow("Frame", frame)
	out.write(frame)                      
	print("%d th frame finished" %count)
    
	#FPS = 3
	cv2.waitKey(500)
    
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

print(label)        
cap.release()
out.release()                    

# do a bit of cleanup
cv2.destroyAllWindows()