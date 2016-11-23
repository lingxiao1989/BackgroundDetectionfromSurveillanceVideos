import cv2,numpy as np
import matplotlib.pyplot as plt
import time

def reject_outliers(index,frames):
	data=[]
	for img in frames:
		data.append(img[index])
	result=np.array(data)
	return np.mean(result[abs(result - np.mean(result)) <  np.std(result)])

def average(index,frames):
	data=[]
	for img in frames:
		data.append(img[index])
	result=np.array(data)
	return np.mean(result)
		
np.set_printoptions(threshold=np.nan)
vidcap = cv2.VideoCapture('My Movie.mp4')
frames=[]
count = 0
success = True
while (count<7):
  vidcap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,count*1000) 
  success,image = vidcap.read()
  print 'Read a new frame: ', success
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  frames.append(gray)
  cv2.imwrite("frame%d.jpg" % count, gray)
  count += 1

i = 0
diffcount=np.zeros(frames[0].shape)

while (i<count):
	pre=np.array(frames[i])
	i+=1
	j=i
	while(j<count):
		diff= np.array(frames[j]-frames[i-1])
		diff[diff>205]=0
		diff[diff<50]=0
		diff[diff!=0]=1
		diffcount+=diff
		j += 1

background= np.array(frames[0])
for index, x in np.ndenumerate(diffcount):
	if x>0:
		background[index]=reject_outliers(index, frames)
	else:
		background[index]=average(index, frames)
		

cv2.imwrite("reconstimg.jpg",background)
