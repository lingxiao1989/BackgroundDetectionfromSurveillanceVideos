import cv2,numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import scipy.stats

#from PIL import Image

def Gaussion_filter(index,frames,background):
	data=[]
	weighted_prob_addup=0
	prob_addup=0
	for img in frames:
		data.append(img[index])
	array=np.array(data)
	for x in np.nditer(array):
		#print background[index],x
		weighted_prob_addup+=scipy.stats.norm(background[index],100).pdf(x)*x
		prob_addup+=scipy.stats.norm(background[index],100).pdf(x)
		#print background[index],x,scipy.stats.norm(background[index],100).pdf(x)*x,weighted_prob_addup,prob_addup,weighted_prob_addup/prob_addup
	#print result
	#return np.mean(result[abs(result - np.mean(result)) <  np.std(result)])
	return weighted_prob_addup/prob_addup

def background_shaper(index,frames,background):
	data=[]
	weighted_prob_addup=0
	prob_addup=0
	for img in frames:
		data.append(img[index])
	array=np.array(data)
	for x in np.nditer(array):
		pro=scipy.stats.norm(background[index],100).pdf(x)
		
		weighted_prob_addup+=pro*x
		prob_addup+=pro
	return weighted_prob_addup/prob_addup
	
	
np.set_printoptions(threshold=np.nan)
#f1=open('./testfile', 'w+')
vidcap = cv2.VideoCapture('Surveillance Feed - Parking Lot.mp4')
frames=[]
count = 0
success = True
while (count<3):
  vidcap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,count*3000) 
  success,image = vidcap.read()
  print 'Read a new frame: ', success
  #with open('frames.pkl', 'wb') as output:
	#cPickle.dump(image, output, cPickle.HIGHEST_PROTOCOL)
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

iterations=0
background= np.array(frames[0])
while (iterations<4):
	for index, x in np.ndenumerate(diffcount):
		if x>0:
			background[index]=Gaussion_filter(index, frames, background)
	print iterations
	cv2.imwrite("reconsting%d.jpg" % iterations,background)
	iterations+=1

#for index, x in np.ndenumerate(diffcount):
	#if x>0:
		#background[index]=background_shaper(index, frames, background)
#cv2.imwrite("reconsting_final.jpg",background)
	
#cv2.imwrite("reconstimg.jpg",background)
#a= np.array(frames[2]-frames[1])
#a[a>205]=0
#a[a<50]=0
#a[a!=0]=255
#np.savetxt("./testfile",diffcount)
#U1, sigma1, V1 = np.linalg.svd(frames[0])
#U2, sigma2, V2 = np.linalg.svd(frames[1])
#U3, sigma3, V3 = np.linalg.svd(frames[2])
#U, sigma, V = np.linalg.svd(image)	
#print np.mean(test)
#background= np.array(frames[0])
#mean=background.flatten()
#cov=np.eye(mean.size)
#reconstimg=np.matrix(U3[:,:240]) * np.diag(sigma1) * np.matrix(V3[:240,:])