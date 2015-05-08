import os
import sys
import time

import numpy as np
import cv2

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dA import dA
from SdA import SdA

from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


def loadDataset():
    
    path = "/home/rik/jpg/"
    ext = '.jpg'

    dataset = []
    for file in range(1, 801):
 
        img = cv2.imread(path + str(file) + ext)
        img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
        image = np.asarray(img, dtype = np.float32)
        
        image1 = np.reshape(image, (28 * 28 * 3))
        
        image1 /= 255.
        dataset.append(image1)

    print "Number of images: ", len(dataset)
   
    dataset = np.asarray(dataset, dtype = np.float32)
    train_set_x = theano.shared(np.asarray(dataset, dtype=theano.config.floatX), borrow=True)
    
    return train_set_x      
   
def loadTestset():
    path = "/home/rik/jpg/"
    ext = '.jpg'
    dataset = []
    for file in range(1, 801):
 
        img = cv2.imread(path + str(file) + ext)
        img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
        image = np.asarray(img, dtype = np.float32)
        
        image1 = np.reshape(image, (1, 28 * 28 * 3))
        
        image1 /= 255.
        dataset.append(image1)

    return dataset

if __name__ == "__main__":
   
    pretraining_epochs = 1000
    pretrain_lr = 0.1
    train_set_x = loadDataset()
    batch_size = 20
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    #theano defines
    x = T.matrix('x')
    index = T.lscalar()
    
    # random generator for weight init
    rng = np.random.RandomState()
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    sda = SdA(numpy_rng=rng,
        theano_rng=theano_rng,
        n_ins=28 * 28 * 3,
        hidden_layers_sizes=[1000, 1000],
        n_outs=1
        )
        
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    
    
    corruption_levels = [.1, .2, .3]
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)
                
   # hid = da.get_hidden_values(x)
   # hidden = theano.function([x], hid) 
     
    rec = sda.reconstruct(x)
    reconstruct = theano.function([x], rec)
    
    dataset = loadTestset()
    
    for x in range(0, 800 - 1):
        orig_image = dataset[x];
        orig_image *= 255.
        orig_image = np.reshape(orig_image, (28,28,3))
        orig_image = np.asarray(orig_image, dtype = np.uint8)
        orig_image = cv2.resize(orig_image, (280, 280), interpolation = cv2.INTER_LINEAR)
        
        cv2.imshow('original', orig_image)
        
        image = reconstruct(dataset[x])
        
        image *= 255.
        image = np.reshape(image, (28,28,3))
        image = np.asarray(image, dtype = np.uint8)
        image = cv2.resize(image, (280, 280), interpolation = cv2.INTER_LINEAR)
        cv2.imshow('reconstruct', image)
        cv2.waitKey(0)
    
    # start-snippet-4
    '''
    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(15, 15),
        tile_spacing=(1, 1)))
    image.save('weight.png')
    '''
    # end-snippet-4
        
