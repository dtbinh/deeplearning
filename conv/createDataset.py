import numpy as np
import os
import cv2

path = "/home/rik/objects/training/"
objects = os.listdir(path)

scaleSize = 56
dbImages = []
dbLabels = []

for segment in objects:
    files =  os.listdir(path + segment)
    
    for f in files:
        image = cv2.imread(path + segment + "/" + f)
    #    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (scaleSize, scaleSize))
        
    #    cv2.imshow("image", image)
    #    cv2.waitKey(1)
        image = np.asarray(image, dtype=np.float32)
        image /= 255 # normalize images
        
        image = np.reshape(image, (scaleSize * scaleSize * 3))
        
        dbImages.append(image)
        dbLabels.append(segment)
        
    
       
np.save('/home/rik/deeplearning/conv/trainImages.npy', dbImages)
np.save('/home/rik/deeplearning/conv/trainLabels.npy', dbLabels)

print len(dbImages), 'training images' 


path = "/home/rik/objects/validation/"
objects = os.listdir(path)

dbImages = []
dbLabels = []

for segment in objects:
    files =  os.listdir(path + segment)
    
    for f in files:
        image = cv2.imread(path + segment + "/" + f)
     #   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (scaleSize, scaleSize))
        
    #    cv2.imshow("image", image)
    #    cv2.waitKey(1)
        image = np.asarray(image, dtype=np.float32)
        image /= 255 # normalize images
        
        image = np.reshape(image, (scaleSize * scaleSize * 3))
        
        dbImages.append(image)
        dbLabels.append(segment)
        
    
       
np.save('/home/rik/deeplearning/conv/validImages.npy', dbImages)
np.save('/home/rik/deeplearning/conv/validLabels.npy', dbLabels)

print len(dbImages), 'validation images' 
