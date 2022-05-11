import numpy as np 
import cv2

# matrix = [ [255] * 600] * 600
matrix = [ [255] * 600,[0] * 600] * 300

# OpenCv use 8bit unsigned integer data type
# 8 bit unsigned integer range - 0 to 255
image = np.array(matrix, dtype=np.uint8)

print(image)

print(type(image))
print(image.shape)

cv2.imshow('xd',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


