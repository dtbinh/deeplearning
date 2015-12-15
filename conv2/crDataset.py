#!/usr/bin/env python

import cPickle
import numpy as np
import cv2

images = []
labels = []

def createImage(x):
	channels = np.array_split(x.astype(np.float32), 3)
	image = cv2.merge((channels[2], channels[1], channels[0])).reshape((3072,1))
	image = np.reshape(image, (32 * 32 * 3))
	image_temp = cv2.resize(image, (32,32))
	image_temp = np.asarray(image_temp, dtype=np.uint8)
	#cv2.imshow("image", image_temp)
	#cv2.waitKey(0)
	image /= 255
	images.append(np.array(image, dtype=np.float32))
	return 1

for i in range(1,4):
	fo = open('cifar-10-batches-py/data_batch_%d' % (i), 'rb')
	dict = cPickle.load(fo)
	fo.close()

	labels += dict['labels']

	np.apply_along_axis(createImage, axis=1, arr=dict['data'])
	
print images
np.save('trainImages.npy', images)
np.save('trainLabels.npy', labels)


images = []
labels = []

def createImage(x):
	channels = np.array_split(x.astype(np.float32), 3)
	image = cv2.merge((channels[0], channels[1], channels[2])).reshape((3072,1))
	image = np.reshape(image, (32 * 32 * 3))
	image /= 255
	images.append(np.array(image, dtype=np.float32))
	return 1


fo = open('cifar-10-batches-py/data_batch_%d' % (5), 'rb')
dict = cPickle.load(fo)
fo.close()

labels += dict['labels']

np.apply_along_axis(createImage, axis=1, arr=dict['data'])
np.save('validImages.npy', images)
np.save('validLabels.npy', labels)

