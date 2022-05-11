#Import library
from traceback import print_tb
import cv2
import numpy as np

#Create trackbar and Result window 
cv2.namedWindow('result')

#trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
	pass

#Hue lower and Upper range
cv2.createTrackbar('l_h','result',0,179,nothing)
cv2.createTrackbar('u_h','result',0,179,nothing)
cv2.setTrackbarPos('u_h','result',179)

#Saturation lower and Upper range
cv2.createTrackbar('l_s','result',0,255,nothing)
cv2.createTrackbar('u_s','result',0,255,nothing)
cv2.setTrackbarPos('u_s','result',255)

#value lower and Upper range
cv2.createTrackbar('l_v','result',0,255,nothing)
cv2.createTrackbar('u_v','result',0,255,nothing)
cv2.setTrackbarPos('u_v','result',255)

#Cature video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Take each frame
    _, frame = cap.read()
    
    #resize frame
    frame  = cv2.resize(frame,None,fx=.9,fy=.9)
    print(frame.shape)
    
    main_frame = frame.copy()
    
    #add font in main_frame frame
    main_frame = cv2.putText(main_frame,'Main Frame',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Get trackbar value 
    l_h= int(cv2.getTrackbarPos('l_h','result'))
    u_h= int(cv2.getTrackbarPos('u_h','result'))
    
    l_s= int(cv2.getTrackbarPos('l_s','result'))
    u_s= int(cv2.getTrackbarPos('u_s','result'))
    
    l_v= int(cv2.getTrackbarPos('l_v','result'))
    u_v= int(cv2.getTrackbarPos('u_v','result'))
    
    #Make lower and upper range np array
    lower_range_color = np.array([l_h,l_s,l_v])
    upper_range_color = np.array([u_h,u_s,u_v])
    
    #Threshold the HSV image to get defined colors
    mask = cv2.inRange(hsv, lower_range_color, upper_range_color)
    mask_copy = mask.copy()
    
    # Get black portion index in mask and make Zero in same index
    indx = np.where(mask != 255)
    frame[indx] = 0
    result = frame
    
    #add font in result frame
    result = cv2.putText(result,'Result',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
    #conver mask into 3 channel
    mask_copy = cv2.cvtColor(mask_copy,cv2.COLOR_GRAY2BGR)
    
    #add font in mask_copy frame
    mask_copy = cv2.putText(mask_copy,'Mask',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
    result = np.concatenate((main_frame,mask_copy,result),axis=1)
    cv2.imshow('result',result)
    
    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break
    
cv2.destroyAllWindows()
    
    
    