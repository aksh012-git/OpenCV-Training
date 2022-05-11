import numpy as np 
import cv2

# matrix = [[[254,200,222]]*300]*300
matrix = [[[254,200,222]]*300,[[100,250,200]]*300]*150

image = np.array(matrix, dtype=np.uint8)

print(image)

print(type(image))
print(image.shape)


cv2.imshow('xd',image)
cv2.waitKey(0)
cv2.destroyAllWindows()