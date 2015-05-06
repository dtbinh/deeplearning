import os
import sys
import time

import numpy as np
import cv2

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dA import dA

from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


def loadDataset():
    
    path = "/home/rik/testImages/"
    ext = '.jpg'

    dataset = []
    for file in range(1, 6):
 
        img = cv2.imread(path + str(file) + ext, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        
        image = np.asarray(img, dtype = np.float32)
        
        image1 = np.reshape(image, (28 * 28))
        
        image1 /= 255.
        dataset.append(image1)

    print "Number of images: ", len(dataset)
   
    dataset = np.asarray(dataset, dtype = np.float32)
    train_set_x = theano.shared(np.asarray(dataset, dtype=theano.config.floatX), borrow=True)
    
    return train_set_x      
   
def loadTestset():
    path = "/home/rik/testImages/"
    ext = '.jpg'
    dataset = []
    for file in range(1, 6):
 
        img = cv2.imread(path + str(file) + ext, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        
        image = np.asarray(img, dtype = np.float32)
        
        image1 = np.reshape(image, (1, 28 * 28))
        
        image1 /= 255.
        dataset.append(image1)

    return dataset

if __name__ == "__main__":
   
    training_epochs = 100
    learning_rate = 0.1
    train_set_x = loadDataset()
    batch_size = 1
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    #theano defines
    x = T.matrix('x')
    index = T.lscalar()
    
    # random generator for weight init
    rng = np.random.RandomState()
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=100)
        
        
    cost, updates = da.get_cost_updates(
        corruption_level=0.1,
        learning_rate=learning_rate
    )
    
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )    
    
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)
     
    
    hid = da.get_hidden_values(x)
    hidden = theano.function([x], hid) 
     
    rec = da.get_reconstructed_input(x)
    reconstruct = theano.function([x], rec)
    
    dataset = loadTestset()
    
    for x in range(0, 5):
        orig_image = dataset[x];
        orig_image *= 255.
        orig_image = np.reshape(orig_image, (28,28))
        cv2.imshow('original', orig_image)
        image = reconstruct(hidden(dataset[x]))
        
        image *= 255.
        image = np.reshape(image, (28,28))
        
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
        