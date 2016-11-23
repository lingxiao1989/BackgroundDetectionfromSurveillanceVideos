import cv2,numpy as np
import matplotlib.pyplot as plt
import time
#from PIL import Image

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
#f1=open('./testfile', 'w+')
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
#cv2.imwrite("frame%d.jpg" % count, frames[])     # save frame as JPEG file
#cv2.imwrite("frame3.jpg", frames[1])     # save frame as JPEG file
	pre=np.array(frames[i])
	i+=1
	j=i
	while(j<count):
	#height, width = U.shape
	#print height, width
	#np.savetxt("./testfile%d" % count,U)
		diff= np.array(frames[j]-frames[i-1])
		diff[diff>205]=0
		diff[diff<50]=0
		diff[diff!=0]=1
		diffcount+=diff
		j += 1
		#framesdiff.append(diff)
#print >>f1, frames[1]
#U1, sigma1, V1 = np.linalg.svd(frames[0])
#U2, sigma2, V2 = np.linalg.svd(frames[1])
#U3, sigma3, V3 = np.linalg.svd(frames[2])
background= np.array(frames[0])
for index, x in np.ndenumerate(diffcount):
	if x>0:
		background[index]=reject_outliers(index, frames)
	else:
		background[index]=average(index, frames)
		

#a= np.array(frames[2]-frames[1])
#a[a>205]=0
#a[a<50]=0
#a[a!=0]=255
#np.savetxt("./testfile",diffcount)
cv2.imwrite("reconstimg.jpg",background)
#U, sigma, V = np.linalg.svd(image)	
#print np.mean(test)
#background= np.array(frames[0])
#mean=background.flatten()
#cov=np.eye(mean.size)
#reconstimg=np.matrix(U3[:,:240]) * np.diag(sigma1) * np.matrix(V3[:240,:])
