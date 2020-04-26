import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle




def extractminibatch(kk,minibatch_num,batchsize,data):
    """
    Used to get a batch out of data.
    :param kk: indices that give the order to data
    :param minibatch_num: which partition of data is concerned
    :param batchsize: 
    :return: a batch
    """
    batch_start = minibatch_num *  batchsize
    batch_end = (minibatch_num+1) * batchsize
    n_samples = data.shape[0]
    if (batch_end + batchsize) <= n_samples:
        idx = kk[batch_start:batch_end]
        batch = data[idx]
    else:
        batch = data[kk[batch_start:]]
        
    return batch




def dense(x, n1, n2, name, dropout_keep = 1):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        out_drop = tf.nn.dropout(out, dropout_keep)
        return out_drop




class nn:
    def __init__(self, n_entry, shape_encoder, learning_rate = 0.04, nbr_lay_au = 0):
        self.lr = learning_rate
        self.n_entry = n_entry
        self.shape_encoder = shape_encoder
        self.nbr_lay_au = nbr_lay_au

    def network(self, x, shape, reuse=False): # 
        """
        Encode part of the autoencoder
        :param x: network input
        :param shape: [l1,_,_]
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """
        layers = []
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('network'):
            e_dense_1 = tf.nn.tanh(dense(x, self.n_entry, shape[0], 'e_dense_1', dropout_keep = 0.9))
            layers.append(e_dense_1)
            for a in range(len(shape)-1):
                layer_inte = tf.nn.tanh(dense(layers[-1],shape[a] , shape[a+1], 'e_dense_'+str(a+2), dropout_keep = 0.9))
                layers.append(layer_inte)
            output = tf.nn.sigmoid(dense(layers[-1], shape[-1], 1, 'd_output'))
            return output


    def reconstruct(self,data_x):
        with tf.variable_scope(tf.get_variable_scope()):
            output = self.network(data_x, self.shape_encoder, reuse=True)
        return output

    def train(self, data_x, data_y, data_test_x, batch_size=1, n_epoches=1, train_model = True):

        """
        Used to train the autoencoder by passing in the necessary inputs.
        :param data_x: training data
        :param data_test_x: test data
        :return: does not return anything
        """
        
        n_data = data_x.shape[0]

        # Placeholders for input data and the targets
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_entry], name='Input')
        noise = tf.random_normal(tf.shape(x_input), mean  = 0, stddev = 0.0, dtype = tf.float32)
        x_input_noised = x_input + noise
        y_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Target')

        

        with tf.variable_scope(tf.get_variable_scope()):
            output = self.network(x_input_noised, self.shape_encoder)

        y_re = self.reconstruct(x_input)

        # Loss
        
        noise_t = tf.random_normal(tf.shape(y_target), mean  = 0, stddev = 0.0, dtype = tf.float32)
        y_target_noised = y_target + noise_t
        loss = tf.reduce_mean(tf.square(y_target_noised - output))


        # Optimizer
##        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(loss)
##        optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum = 0.1).minimize(loss)
        init = tf.global_variables_initializer()

        n_batches = n_data // batch_size
            
        if n_batches == 0:
            n_batches = 1

        # deep copy
        data_x_cpy = data_x.copy()
        data_y_cpy = data_y.copy()
        inds = np.arange(n_data)
        
        # Saving the model
        saver = tf.train.Saver()
        step = 0
        errs = []
        with tf.Session() as sess:
            sess.run(init)
            if train_model:
                for epoch in range(n_epoches):
                    np.random.shuffle(inds)
                    mean_cost = []
##                    sess.run(init)
                    for b in range(n_batches):
                        batch_x = extractminibatch(inds,b,batch_size,data_x_cpy)
                        batch_y = extractminibatch(inds,b,batch_size,data_y_cpy)
                        sess.run(optimizer, feed_dict={x_input: batch_x, y_target: batch_y})
                        cost = sess.run(loss,feed_dict={x_input:batch_x, y_target: batch_y})
                        mean_cost.append(cost)
                        step += 1
                    errs.append(np.mean(mean_cost))
                    print('Epoch %d Cost %g' % (epoch, errs[-1]))
                print("Model Trained!")
                saver.save(sess, save_path='./test.ckpt')
            # test
            self.test = sess.run(y_re,feed_dict = {x_input:data_test_x})


            


