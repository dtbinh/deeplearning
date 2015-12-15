import numpy as np
import os
import cv2

import sys
import time

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from numpy import broadcast

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """
    
    rng = np.random.RandomState()
    srng = RandomStreams(rng.randint(999999))
    
    def drop(self, input, p=0.5, rng=rng): 
        """
        :type input: numpy.array
        :param input: layer or weight matrix on which dropout resp. dropconnect is applied
        
        :type p: float or double between 0. and 1. 
        :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
        
        """   
            
        mask = self.srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
        return input * mask

    def __init__(self, rng, input, is_train, filter_shape, image_shape, poolsize=(2, 2), p=0.5):
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
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        ) 
        
        conv_out_train = self.drop(np.cast[theano.config.floatX](1./p) * conv_out, p=p)        
     #   out = conv_out;
        out =  T.switch(T.neq(is_train, 0), conv_out_train, conv_out)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=out,
            ds=poolsize,
            ignore_border=True
        )
        
     #   pooled_out_train = self.drop(np.cast[theano.config.floatX](1./p) * pooled_out, p=p)
     #   pooled_out_drop = T.switch(T.neq(is_train, 0), pooled_out_train, pooled_out)
        pooled_out_drop = pooled_out

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out_drop + self.b.dimshuffle('x', 0, 'x', 'x'))
        

        # store parameters of this layer
        self.params = [self.W, self.b]


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
        labels.append(x)
       
    return images, labels

def load_valid_data(path):
    images = np.load(path + 'validImages.npy')
    labelNames = np.load(path + 'validLabels.npy')
    
    labels = []
    
    for x in labelNames:
        labels.append(x)

    return images, labels

if __name__ == "__main__":
    
    batch_size = 300
    lr = 0.1   # starting learning rate
    lowest_lr = 0.001
    lr_decay_rate = 0.95
    initial_momentum = 0.5  # starting momentum
    momentum_decay_rate = 0.98
    p = 0.5     # prop. of NOT dropping a neuron
    nkerns=[16, 20, 20]
    n_epochs=1500
    
    is_train = T.iscalar('is_train')
    
    learning_rate = theano.shared(np.asarray(lr,
        dtype=theano.config.floatX))
    
    assert initial_momentum >= 0. and initial_momentum < 1.
    momentum =theano.shared(np.cast[theano.config.floatX](initial_momentum), name='momentum')
    
    
    scaleSize = 32
    
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
    layer0_input = x.reshape((-1, 3, scaleSize, scaleSize))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, scaleSize, scaleSize),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2),
        is_train=is_train,
        p=p
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    
    scale2Size = ((scaleSize - 4) / 2)
    
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], scale2Size, scale2Size),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2),
        is_train=is_train,
        p=p
    )
    
    scale3Size = ((scale2Size - 4) / 2)
    
    '''
    layer1_extra = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], scale3Size, scale3Size),
        filter_shape=(nkerns[2], nkerns[1], 5, 5),
        poolsize=(2, 2),
        is_train=is_train,
        p=p
    )
    '''
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
 #   layer2_input = layer1_extra.output.flatten(2)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * scale3Size * scale3Size,
        n_out=4000,
        activation=T.nnet.relu,
        is_train=is_train,
        p=p
    )
    
    layer22 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4000,
        n_out=4000,
        activation=T.tanh,
        is_train=is_train,
        p=p
    )
    
    layer222 = HiddenLayer(
        rng,
        input=layer22.output,
        n_in=4000,
        n_out=100,
        activation=T.tanh,
        is_train=is_train,
        p=p
    )
    
   
    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer222.output, n_in=100, n_out=10)

    
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

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

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

    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1        
                
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index


            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                if epoch%10 == 0:
                    print "momentum: ", momentum.get_value()
                    print "learing rate: ", learning_rate.get_value()
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

        if momentum.get_value() < 0.99:
            new_momentum = 1. - (1. - momentum.get_value()) * momentum_decay_rate
            momentum.set_value(np.cast[theano.config.floatX](new_momentum))
        
        if (learning_rate.get_value() > lowest_lr):
            new_learning_rate = learning_rate.get_value() * lr_decay_rate
            
            if new_learning_rate < lowest_lr:
                new_learning_rate = lowest_lr
            
            learning_rate.set_value(np.cast[theano.config.floatX](new_learning_rate))
         
                
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
    
