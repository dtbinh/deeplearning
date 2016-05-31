import numpy as np
import os
import cv2

path = "/home/rik/objects/training/"
objects = os.listdir(path)

scaleSize = 32
dbImages = []
dbLabels = []

for segment in objects:
    files =  os.listdir(path + segment)
    
    for f in files:
        image = cv2.imread(path + segment + "/" + f)
        image = cv2.resize(image, (scaleSize, scaleSize))
        r,g,b = cv2.split(image)
        imgNew = [r, g, b]
    #    cv2.imshow("image", image)
    #    cv2.waitKey(1)
        image = np.asarray(imgNew, dtype=np.float32)
        image /= 255 # normalize images
        image = np.reshape(image, (scaleSize * scaleSize * 3))
        
        dbImages.append(image)
        dbLabels.append(segment)
        
  
       
np.save('/home/rik/deeplearning/conv2/trainImages.npy', dbImages)
np.save('/home/rik/deeplearning/conv2/trainLabels.npy', dbLabels)

print len(dbImages), 'training images' 


path = "/home/rik/objects/validation/"
objects = os.listdir(path)

dbImages = []
dbLabels = []

for segment in objects:
    files =  os.listdir(path + segment)
    
    for f in files:
        image = cv2.imread(path + segment + "/" + f)
        image = cv2.resize(image, (scaleSize, scaleSize))
        r,g,b = cv2.split(image)
        imgNew = [r, g, b]
    #    cv2.imshow("image", image)
    #    cv2.waitKey(1)
        image = np.asarray(imgNew, dtype=np.float32)
        image /= 255 # normalize images
        image = np.reshape(image, (scaleSize * scaleSize * 3))
        
        dbImages.append(image)
        dbLabels.append(segment)
        
    
       
np.save('/home/rik/deeplearning/conv2/validImages.npy', dbImages)
np.save('/home/rik/deeplearning/conv2/validLabels.npy', dbLabels)

print len(dbImages), 'validation images' 
