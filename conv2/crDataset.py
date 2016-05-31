#!/usr/bin/env python

import cPickle
import numpy as np
import cv2

images = []
labels = []

def createImage(x):
	channels = np.array_split(x.astype(np.float32), 3)
	image = cv2.merge((channels[0], channels[1], channels[2])).reshape((3072,1))
	image = np.reshape(image, (32 * 32 * 3))
	image /= 255
	images.append(np.array(image, dtype=np.float32))
	return 1

for i in range(1,6):
	fo = open('cifar-10-batches-py/data_batch_%d' % (i), 'rb')
	dict = cPickle.load(fo)
	fo.close()

	labels += dict['labels']

	np.apply_along_axis(createImage, axis=1, arr=dict['data'])
	

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


fo = open('cifar-10-batches-py/test_batch', 'rb')
dict = cPickle.load(fo)
fo.close()

labels += dict['labels']

np.apply_along_axis(createImage, axis=1, arr=dict['data'])
np.save('validImages.npy', images)
np.save('validLabels.npy', labels)

