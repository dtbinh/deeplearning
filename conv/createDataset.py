import numpy as np
import os
import cv2

path = "/home/rik/objects/"
objects = os.listdir(path)

scaleSize = 32
dbImages = []
dbLabels = []

for segment in objects:
    files =  os.listdir(path + segment)
    
    for f in files:
        image = cv2.imread(path + segment + "/" + f)
        image = cv2.resize(image, (scaleSize, scaleSize))
        
        cv2.imshow("image", image)
        cv2.waitKey(1)
        image = np.asarray(image, dtype=np.float32)
        image /= 255 # normalize images
        
        image = np.reshape(image, (scaleSize * scaleSize * 3))
        
        dbImages.append(image)
        dbLabels.append(segment)
        
    
       
np.save('/home/rik/deeplearning/conv/objectImages.npy', dbImages)
np.save('/home/rik/deeplearning/conv/objectLabels.npy', dbLabels)

