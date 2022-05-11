# import cv2

# image = cv2.imread('/home/wot-aksh/Desktop/Python_Training/19_opencv/road.jpg')

# image = cv2.resize(image,(800,545))

# output = image.copy()

# cv2.imshow('jnj1',image)

# cv2.rectangle(output, (420, 224),
# 						(455, 270), (258,255, 255), 2)

# cv2.putText(output, 'Truck detected', (420, 220), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# cv2.imshow('jnj',output)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ------------------------------------------------------------------------------------------------------------


# import cv2

# img1 = cv2.imread('/home/wot-aksh/Desktop/Python_Training/19_opencv/image/photo-1593005510329-8a4035a7238f.jpeg') 
# img2 = cv2.imread('/home/wot-aksh/Desktop/Python_Training/19_opencv/image/download.png')

# img2 = cv2.resize(img2,(600, 400))
# img1 = cv2.resize(img1,(600, 400))

# cv2.imshow('Bitwise And1', img1)
# cv2.imshow('Bitwise And2', img2)

# print(img1.shape,img2.shape)

# dest_and = cv2.bitwise_and(img2, img1, mask = None)
 
# cv2.imshow('Bitwise And', dest_and)
  
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()





# import numpy as np
# import cv2 as cv

# img = cv.imread('/home/wot-aksh/Desktop/Python_Training/19_opencv/image/road.jpg',1)

# img = cv.resize(img,(800,574))


# cv.imshow('img2',img)

# rows,cols = img.shape[:2]

# print(img.shape)

# pts1 = np.float32([[50,100],[200,50],[50,200]])
# pts2 = np.float32([[10,100],[200,50],[100,250]])

# M = cv.getAffineTransform(pts1,pts2)

# dst = cv.warpAffine(img,M,(cols,rows))


# cv.imshow('img1',dst)
# cv.waitKey(0)
# cv.destroyAllWindows()



import cv2 as cv
import numpy as np

#trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
	pass

#create a seperate window named 'controls' for trackbar
cv.namedWindow('controls')
#create trackbar in 'controls' window with name 'r''
cv.createTrackbar('l_h','controls',0,179,nothing)
cv.createTrackbar('u_h','controls',0,179,nothing)

cv.createTrackbar('l_s','controls',0,255,nothing)
cv.createTrackbar('u_s','controls',0,255,nothing)

cv.createTrackbar('l_v','controls',0,255,nothing)
cv.createTrackbar('u_v','controls',0,255,nothing)



#cap = cv.imread('/home/wot-aksh/Desktop/Python_Training/19_opencv/image/frame.jpg',1)
cap = cv.VideoCapture(0)

while (1):
    # Take each frame
    _, frame = cap.read()

    print(frame.shape)

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    print(f'{hsv.shape=}')
    
    
    l_h= int(cv.getTrackbarPos('l_h','controls'))
    u_h= int(cv.getTrackbarPos('u_h','controls'))
    
    l_s= int(cv.getTrackbarPos('l_s','controls'))
    u_s= int(cv.getTrackbarPos('u_s','controls'))
    
    l_v= int(cv.getTrackbarPos('l_v','controls'))
    u_v= int(cv.getTrackbarPos('u_v','controls'))
    cv.imshow('controls',frame)
    
    print(f'{l_h=},  {u_h =}')
    # define range of blue color in HSV
    lower_blue = np.array([l_h,l_s,l_v])
    upper_blue = np.array([u_h,u_s,u_v])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    #print(mask.shape)
    cv.imshow('mask',mask)

    # Bitwise-AND mask and original image
    res1 = cv.bitwise_and(frame, frame, mask= None)
    cv.imshow('res1',res1)

    # mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    # res1 = cv.bitwise_and(frame,mask)
    # cv.imshow('res1',res1)

    indx = np.where(mask != 255)
    frame[indx] = 0
    indx = np.flip(np.array(indx).T, axis=1)

    #print(f"{indx = }")
    res2 = frame

    # res2 = frame & mask
    cv.imshow('res2',res2)

    k = cv.waitKey(1)
    if k==27:
        break
    
cv.destroyAllWindows()
