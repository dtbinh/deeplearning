import cPickle
import numpy as np
import cv2

img = []
labels = []

for i in range(1, 6):
    fo = open('cifar-10-batches-py/data_batch_%d' % (i), 'rb')
    dict = cPickle.load(fo)
    fo.close()
    
    images = dict['data']
    images = np.asarray(images, dtype=np.float32)
    images /= 255.
    
    for i in range(0, len(images)):
        img.append(np.reshape(images[i], (32*32*3), order='F'))        
    
    labels += dict['labels']
        

#img = cv2.resize(trainImages[0], (32,32))
#cv2.imshow("image", img)
#cv2.waitKey(0)
#trainImages = np.vstack(trainImages)
print len(img[0])
print len(img)
print len(labels)
img = np.asarray(img, dtype=np.float32)


np.save('trainImages.npy', img)
np.save('trainLabels.npy', labels)

testImages = []
labels = []
img = []

fo = open('cifar-10-batches-py/test_batch', 'rb')
dict = cPickle.load(fo)
fo.close()

images = dict['data']
images = np.asarray(images, dtype=np.float32)
images /= 255.

for i in range(0, len(images)):
    img.append(np.reshape(images[i], (32*32*3), order='F'))       

labels += dict['labels']

img = np.asarray(img, dtype=np.float32)

np.save('validImages.npy', img)
np.save('validLabels.npy', labels)
