import numpy as np
import os
import cv2

import sys
import time

import matplotlib.pyplot as plt

import theano
import theano.tensor as T
#from theano.tensor.signal import downsample
#from theano.tensor.nnet import conv
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from numpy import broadcast

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """
    rng = np.random.RandomState()
    srng = RandomStreams(rng.randint(9999999))
    
    def drop(self, input, p=0.5, rng=rng): 
        """
        :type input: numpy.array
        :param input: layer or weight matrix on which dropout resp. dropconnect is applied
        
        :type p: float or double between 0. and 1. 
        :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
        
        """   
            
        mask = self.srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
        return input * mask

    def __init__(self, rng, input, is_train, filter_shape, image_shape, padding, stride = 1, poolsize=(2, 2), p= 0.5):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        '''
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        '''
        input_shuffled = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        conv_op = FilterActs(stride=stride, partial_sum=1, pad = padding)
        contiguous_input = gpu_contiguous(input_shuffled)
        contiguous_filters = gpu_contiguous(filters_shuffled)
        conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)
        
               
        if poolsize != (-1, -1):
            pool_op = MaxPool(ds=poolsize[0], stride=poolsize[0])
            pooled_out_shuffled = pool_op(conv_out_shuffled)
            pooled_out = pooled_out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01
            
            output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            output = T.nnet.relu(conv_out_shuffled.dimshuffle(3, 0, 1, 2) + self.b.dimshuffle('x', 0, 'x', 'x'))


        train_output = self.drop(np.cast[theano.config.floatX](1./p) * output, p=p)
       
        self.output = T.switch(T.neq(is_train, 0), train_output, output)
        # store parameters of this layer
        self.params = [self.W, self.b]
        
        
        def getW():
            return self.W
        


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')


def load_train_data(path):
    images = np.load(path + 'trainImages.npy')
    labelNames = np.load(path + 'trainLabels.npy')
    #print labelNames 
    labels = []
    
    for x in labelNames:
        if x == "aaDrink":
            l = 0
        if x == "chickenSoup":
            l = 1
        if x == "cocaColaLight":
            l = 2
        if x == "fanta":
            l = 3
        if x == "pringles":
            l = 4
        if x == "redBull":
            l = 5
        if x == "shampoo":
            l = 6
        if x == "yellowContainer":
            l = 7
        
        labels.append(l)
       
    return images, labels

def load_valid_data(path):
    images = np.load(path + 'validImages.npy')
    labelNames = np.load(path + 'validLabels.npy')
    
    labels = []
    
    for x in labelNames:
        if x == "aaDrink":
            l = 0
        if x == "chickenSoup":
            l = 1
        if x == "cocaColaLight":
            l = 2
        if x == "fanta":
            l = 3
        if x == "pringles":
            l = 4
        if x == "redBull":
            l = 5
        if x == "shampoo":
            l = 6
        if x == "yellowContainer":
            l = 7
        
        labels.append(l)

    return images, labels

if __name__ == "__main__":
    
    batch_size = 128 * 1   # algo is optimized for multiple of 128
    lr = 0.01   # starting learning rate
    lowest_lr = 0.0001
    lr_decay_rate = 0.995
    initial_momentum = 0.5  # starting momentum
    momentum_decay_rate = 0.98
    p = 0.5
    nkerns=[16, 16, 256]  # only multiple of 16 kernels allowed
    n_epochs=1500
    # DON'T FORGET TO ADD NEW LAYERS TO PARAMS!
    

    
    os.system('rm ./kernels/*')  # Remove all the files in the kernels map
    
    
    is_train = T.iscalar('is_train')
    
    learning_rate = theano.shared(np.asarray(lr,
        dtype=theano.config.floatX))
    
    assert initial_momentum >= 0. and initial_momentum < 1.
    momentum =theano.shared(np.cast[theano.config.floatX](initial_momentum), name='momentum')
    
    
    scaleSize = 100
    
    path = "/home/rik/deeplearning/conv2/"
    train_images, train_labels = load_train_data(path)
    valid_images, valid_labels = load_valid_data(path)
      
    rng = np.random.RandomState(np.random.randint(2*10))

    train_set_x, train_set_y = shared_dataset((train_images, train_labels))
    valid_set_x, valid_set_y = shared_dataset((valid_images, valid_labels))
    test_set_x, test_set_y = shared_dataset((valid_images, valid_labels))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    conv0_input = x.reshape((-1, 3, scaleSize, scaleSize))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    conv0 = LeNetConvPoolLayer(
        rng,
        input=conv0_input,
        image_shape=(batch_size, 3, scaleSize, scaleSize),
        filter_shape=(nkerns[0], 3, 11, 11),
        poolsize=(2, 2), # use (-1,-1) for no pooling
        padding = 5,  # add 0-padding on outside of image
        stride = 1,
        is_train=is_train,
        p=0.5
    )
    '''
    conv00 = LeNetConvPoolLayer(
        rng,
        input=conv0.output,
        image_shape=(batch_size, nkerns[0], scaleSize, scaleSize),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2), # use (-1,-1) for no pooling
        padding = 1,  # add 0-padding on outside of image
        stride = 1,
        is_train=is_train,
        p=1.0
    )
        
  
    conv1 = LeNetConvPoolLayer(
        rng,
        input=conv00.output,
        image_shape=(batch_size, nkerns[1], scaleSize/2, scaleSize/2),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(2, 2),
        padding = 1,
        stride = 1,
        is_train=is_train,
        p=1.0
    )
    '''   
 #   layer2_input = conv1.output.flatten(2)
    layer2_input = conv0.output.flatten(2)
    
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[0] * scaleSize / 2 * scaleSize / 2,
        n_out=8000,
        activation=T.nnet.relu,
        is_train=is_train,
        p=p
    )
    
    layer22 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=8000,
        n_out=2000,
        activation=T.nnet.relu,
        is_train=is_train,
        p=p
    )
    
    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer22.output, n_in=2000, n_out=8)

    
    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: np.cast['int32'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: np.cast['int32'](0)
        }
    )
    
    training_error_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: np.cast['int32'](0)
            }
    )

    # create a list of all model parameters to be fit by gradient descent
 #   params = layer3.params + layer22.params + layer2.params + conv1.params + conv00.params + conv0.params
    params = layer3.params + layer22.params + layer2.params + conv0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    
    '''
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
    '''
    
    updates = []
    for param in  params:
        param_update = theano.shared(param.get_value()*np.cast[theano.config.floatX](0.))    
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (np.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))  
    

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            is_train: np.cast['int32'](1)
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = batch_size * n_epochs  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    plt.figure(1)
    plt.show(block=False)
    plt.ion()
    
    plt.title('Training Error')
    plt.axis([0,n_epochs / 10, 0, 100])
    
    plt.figure(2)
    plt.show(block=False)
    plt.ion()
    
    plt.title('Validation Error')
    plt.axis([0,n_epochs / 10, 0, 100])
    
    trainError = []
    validError = []
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1        
                
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                  
                 if epoch%10 == 0:
                # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    
                    training_error = [training_error_model(i) for i 
                                      in xrange(n_train_batches)]
                    this_training_loss = np.mean(training_error)
               
                    print "momentum: ", momentum.get_value()
                    print "learing rate: ", learning_rate.get_value()
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))
                    print('epoch %i, minibatch %i/%i, training error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_training_loss * 100.))
                    
                    
                    trainError.append(this_training_loss * 100.)
                    validError.append(this_validation_loss * 100.)
                    
                    plt.figure(1)
                #    plt.clf()
                    plt.plot(trainError)
                    plt.draw()
                    
                    plt.figure(2)
                #    plt.clf()
                    plt.plot(validError)
                    plt.draw()
                    
                    for i in range(0, nkerns[0]):
                        data = conv0.W.get_value()[i]
                        max_scale = np.max(data)
                        min_scale = np.min(data)
                        img = (255 * (data - min_scale) / (max_scale - min_scale)).astype('uint8')
                        
                        R = img[0]
                        G = img[1]
                        B = img[2]
                        
                        
                        img = cv2.merge((R,G,B))
                        img = cv2.resize(img, (100,100))
                        cv2.imwrite('kernels/kernel' + str(i) + '.jpg', img)
                        
                        if epoch == 10:
                           cv2.imwrite('kernels/kernel' + str(i) + '_first.jpg', img) 
                    '''
                    for i in range(0, nkerns[0]):
                        dataR = conv0.W.get_value()[i][0]
                        dataG = conv0.W.get_value()[i][1]
                        dataB = conv0.W.get_value()[i][2]
                     
                        dataRmin = np.min(dataR)
                        dataRmax = np.max(dataR) + -dataRmin
                        dataR += -dataRmin
                        dataR /= dataRmax
                        dataR *= 255
                        
                        dataGmin = np.min(dataG)
                        dataGmax = np.max(dataG) + -dataGmin
                        dataG += -dataGmin
                        dataG /= dataGmax
                        dataG *= 255
                        
                        dataBmin = np.min(dataB)
                        dataBmax = np.max(dataB) + -dataBmin
                        dataB += -dataBmin
                        dataB /= dataBmax
                        dataB *= 255
                        
                       # dataR = dataR - np.min(dataR) *  1 / (np.max(dataR) - np.min(dataR)) 
                       # dataG = dataG - np.min(dataG) *  1 / (np.max(dataG) - np.min(dataG)) 
                       # dataB = dataB - np.min(dataB) *  1 / (np.max(dataB) - np.min(dataB)) 
                        img =  cv2.merge((dataR * 255,dataB * 255,dataG * 255))
                        cv2.imwrite('kernels/kernel' + str(i) + '.jpg', img)
                    '''

                    

        if momentum.get_value() < 0.99:
            new_momentum = 1. - (1. - momentum.get_value()) * momentum_decay_rate
            momentum.set_value(np.cast[theano.config.floatX](new_momentum))
        
        if (learning_rate.get_value() > lowest_lr):
            new_learning_rate = learning_rate.get_value() * lr_decay_rate
            
            if new_learning_rate < lowest_lr:
                new_learning_rate = lowest_lr
            
            learning_rate.set_value(np.cast[theano.config.floatX](new_learning_rate))
         
    f = open('training', 'wb')
    plt.figure(1)
    plt.savefig(f)
    f.close()
    f = open('validation', 'wb')
    plt.figure(2)
    plt.savefig(f)
    f.close()
    
              
    '''
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    if epoch%10==0:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
    '''

    end_time = time.clock()
    print ' ran for %.2fm' % ((end_time - start_time) / 60.)
    '''
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    '''

    '''
    
    run_forward = theano.function(inputs = [x], outputs=layer3.p_y_given_x)
    
    for idx in range(0, len(valid_images)):
#    for idx in range(0, len(train_images)):   
        img = valid_images[idx]
        #img = train_images[idx]
        img = np.asarray(img, dtype=np.float32)
        img = np.reshape(img, (scaleSize,scaleSize,3))
      #  img = cv2.resize(img, (320,320))
        cv2.imshow('image', img)
        
        input_data = []
        for i in range(0, batch_size):
            input_data.append(valid_images[idx])
        #    input_data.append(train_images[idx])
        output =  run_forward(input_data)
        
        print "aaDrink:", output[0][0], ", cherryCoke:", output[0][1], ", chickenSoup:", output[0][2], ", cocoColaLight:", output[0][3], ", fanta:", output[0][4]
        print ", hertogJan:", output[0][5], "pringles:", output[0][6], ", redBull:", output[0][7], ", shampoo:", output[0][8], ", yellowContainer:", output[0][9]
        print '**********************************************************************************************************************************************'
        print '\n'
        cv2.waitKey(0)
        
    '''
    
