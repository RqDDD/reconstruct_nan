import tensorflow as tf
import numpy as np
import random as rd
import math
import os
import matplotlib.pyplot as plt


def extractminibatch(kk,minibatch_num,batchsize,data):
    batch_start = minibatch_num *  batchsize
    batch_end = (minibatch_num+1) * batchsize
    n_samples = data.shape[0]
    if (batch_end + batchsize) <= n_samples:
        idx = kk[batch_start:batch_end]
        batch = data[idx]
    else:
        batch = data[kk[batch_start:]]
        
    return batch


def weight_init(shape,mean = 0.0,stddev = 0.1,seed = 1,name='weights',init_method='gauss'):
    
    ''' Not so trivial. Influence on the final output. Maybe to check '''
    
    if init_method.lower() == 'gauss':
        return tf.Variable(tf.truncated_normal(shape=shape,
                                               mean=mean,
                                               stddev=stddev,
                                               seed = seed),
                            name = name)
    elif init_method.lower() == 'uniform':
        low = -1 * np.sqrt(6.0/np.sum(shape))
        high = 1 * np.sqrt(6.0/np.sum(shape))
        return tf.Variable(tf.random_uniform(shape = shape,
                                             minval = low,
                                             maxval = high,dtype = tf.float32))

    else:
        raise ValueError("Error type of init method, should be 'gauss' or 'uniform'")


def bias_init(shape,name='biases'):
    return tf.Variable(tf.constant(rd.random(),shape=shape),name=name)

def leaky_ReLU(feature_in,leaky=0.01,name="Leaky_ReLU"):
    return tf.maximum(leaky*feature_in,feature_in)

def active_function(activation = "relu"):
    if activation.lower() == "relu":
        return tf.nn.relu
    elif activation.lower() == "sigmoid":
        return tf.nn.sigmoid
    elif activation.lower() == "tanh":
        return tf.nn.tanh
    elif activation.lower() == "softplus":
        return tf.nn.softplus
    elif activation.lower() == "linear":
        return lambda x:x
    elif activation.lower() == "lrelu":
        return leaky_ReLU
    else:
        raise ValueError("Error type of activation function, should be relu(*)/sigmoid/tanh/softplus/linear")

    
def output_function(output_fun = "linear"):
    if output_fun.lower() == "linear":
        return lambda x:x
    elif output_fun.lower() == "softmax":
        return tf.nn.softmax
    elif output_fun.lower() == "sigmoid":
        return tf.nn.sigmoid
    else:
        raise ValueError("Error type of Output function, should be \
                         linear(regression problem) or softmax(classification problem)")
    

def cost_function(cost_fun = "mse"):
    if cost_fun.lower() == "mse":
        # mean squared error (linear regression)
        return tf.losses.mean_squared_error
    elif cost_fun.lower() == "expll":
        # exponential log likelihood (poisson regression)
        return tf.nn.log_poisson_loss
    elif cost_fun.lower() == "xent":
        # cross entropy (binary classification)
        return tf.nn.sigmoid_cross_entropy_with_logits
    elif cost_fun.lower() == "mcxent":
        # multi-class cross entropy (classification)->softmax
        return tf.nn.softmax_cross_entropy_with_logits
    elif cost_fun.lower() == 'class':
        return lambda y,y_: tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)),'float')
    else:
        raise ValueError('Error type of Cost Function, should be \
                         mse(*)/expll/xent/mcxent/class(just for calculate class accuracy)')

def optimizer_fun(optimizer_method = 'sgd'):
    if optimizer_method.lower() == 'sgd':
        return tf.train.MomentumOptimizer
    elif optimizer_method.lower() == 'adam':
        return tf.train.AdamOptimizer
    elif optimizer_method.lower() == 'adagrad':
        return tf.train.AdagradDAOptimizer
    elif optimizer_method.lower() == 'rmsprop':
        return tf.train.RMSPropOptimizer
    else:
        raise ValueError('Error type of Optimizer Function, should be sgd(*)/adam/adagrad/rmsprop')




class rbm_partially_sup:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate = 0.01,
                 momentum = 0.0,
                 rbm_type = 'bbrbm',
                 init_method = 'gauss',
                 CD_k=1,
                 wPenalty = 0.0001,
                 cost_Fun = 'mse',
                 dropout = 0.,
                 relu_hidden = False,
                 relu_visible = False,
                 BP_optimizer_method = 'sgd',
                 BP_activation = "relu",
                 BP_output_fun = "linear",
                 BP_cost_fun = "mse",
                 rbm_name = 'rbm',
                 filepath = "saved_model_rbm"
                 ):
                ## n_out,
        # check parameters
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0,1]')
        assert 0. <= dropout <= 1.
            
        # RBM training parameter
        self.rbm_type = rbm_type
        if rbm_type.lower() == 'gbrbm':
            self.sig = 1.0;
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.CD_k = CD_k
        self.relu_hidden = relu_hidden
        self.relu_visible = relu_visible
        self.lr = learning_rate
        self.momentum = momentum
        self.init_method = init_method
        
        # RBM parameters 
        self.w = weight_init([n_visible,n_hidden],
                        mean=0.,
                        stddev=math.sqrt(0.05/(n_visible+n_hidden)),
                        init_method=init_method)
        self.hidden_bias = bias_init([n_hidden])
        self.visible_bias = bias_init([n_visible])
        
        self.v_w = tf.Variable(tf.zeros([n_visible,n_hidden]),dtype=tf.float32)
        self.v_hb = tf.Variable(tf.zeros(n_hidden),dtype=tf.float32)
        self.v_vb = tf.Variable(tf.zeros(n_visible),dtype=tf.float32)
        
        # output layer part: weights and bias
##        self.outlayer_w = weight_init([n_hidden,n_out],
##                        mean=0.,
##                        stddev=math.sqrt(2.0/(self.n_hidden+n_out)),
##                        init_method=self.init_method)
##        self.outlayer_bias = bias_init([n_out])
        
        self.BP_optimizer_method = optimizer_fun(BP_optimizer_method)
        self.BP_activation_fun = active_function(BP_activation)
        self.BP_output_fun = output_function(BP_output_fun)
        self.BP_cost_fun = cost_function(BP_cost_fun)
        
        # fliping index for computing pseudo likelihood
        self.i = 0
        
        # other parameters for training
        self.dropout = dropout
        self.wPenalty = wPenalty
        self.plot = False
        self.cost_fun = cost_Fun
        
        # path to save and restore model parameters
        if os.path.exists(filepath) is False:
            os.mkdir(filepath)
        self._save_path = filepath + "/" + "./" + rbm_name + ".ckpt"

    def actV2H(self,vis):
        if self.rbm_type.lower() == 'bbrbm':
            h_active = tf.matmul(vis,self.w) + self.hidden_bias
        elif self.rbm_type.lower() == 'gbrbm':
            h_active = tf.matmul(vis/self.sig,self.w) + self.hidden_bias
        else:
            raise TypeError('Error Type of rbm, should be bbrbm or gbrbm!')
            
        if self.relu_hidden is True:
            h_prob = tf.nn.relu(h_active)
        else:
            h_prob = tf.nn.tanh(h_active)
        
        return h_prob

    
    def actH2V(self,hid):
        if self.rbm_type.lower() == 'bbrbm':
            v_active = tf.matmul(hid,tf.transpose(self.w)) + self.visible_bias
        elif self.rbm_type.lower() == 'gbrbm':
            v_active = tf.matmul(hid*self.sig,tf.transpose(self.w)) + self.visible_bias
        else:
            raise TypeError('Error Type of rbm, should be bbrbm or gbrbm!')
        
        if self.relu_visible is True:
            v_prob = tf.nn.relu(v_active)
        else:
            if self.rbm_type.lower() == 'bbrbm':
                v_prob = tf.nn.tanh(v_active)
            elif self.rbm_type.lower() == 'gbrbm':
                v_prob = v_active
            else:
                raise TypeError('Error Type of rbm, should be bbrbm or gbrbm!')
        return v_prob


    def CDk(self,visibles, dropout_keep_prob): #  + target
         
        ''' Modified algorithm ; no gibbs sampling (deterministic), raw output of units is considered'''


        v_sample = visibles
        h0_state = self.actV2H(v_sample)
         
        
##        BP_train = self.partially_supervised_part(visibles,targets)
        
        w_positive = tf.matmul(tf.transpose(visibles),h0_state)

        # add noise to visibles to detect more complex features (and avoid overfitting)
        noise = tf.random_normal(tf.shape(visibles), mean  = 0, stddev = 0.02, dtype = tf.float32)
        visibles_noised = visibles + noise

        h_state = self.actV2H(visibles_noised)
        
        # not gibbs
        for i in range(self.CD_k):
            h_state = tf.nn.dropout(h_state, dropout_keep_prob)
            v_state = self.actH2V(h_state)
            h_state = self.actV2H(v_state)
            
        
        w_negative = tf.matmul(tf.transpose(v_state),h_state)
        
        w_grad = tf.divide(tf.subtract(w_positive,w_negative),tf.to_float(tf.shape(visibles_noised)[0]))
        hb_grad = tf.reduce_mean(h0_state-h_state,0)
        vb_grad = tf.reduce_mean(visibles_noised-v_state,0)
        
        if self.rbm_type.lower() == 'gbrbm':
            w_grad = tf.divide(w_grad,self.sig)
            vb_grad = tf.divide(vb_grad,self.sig**2)
        
        return w_grad,hb_grad,vb_grad,v_state

    def rbm_train(self,visibles,dropout_keep_prob): # add target

        
        w_grad,hb_grad,vb_grad,vstate = self.CDk(visibles,dropout_keep_prob)
        # compute new velocities
        v_w = self.momentum * self.v_w + self.lr * (w_grad - self.wPenalty * self.w)
        v_hb = self.momentum * self.v_hb + self.lr * hb_grad
        v_vb = self.momentum * self.v_vb + self.lr * vb_grad
        
        # update rbm parameters
        update_w = tf.assign(self.w,self.w + v_w)
        update_hb = tf.assign(self.hidden_bias,self.hidden_bias+v_hb)
        update_vb = tf.assign(self.visible_bias,self.visible_bias+v_vb)
        
        # update vlocities
        update_v_w = tf.assign(self.v_w,v_w)
        update_v_hb = tf.assign(self.v_hb,v_hb)
        update_v_vb = tf.assign(self.v_vb,v_vb)
        
        return [update_w,update_hb,update_vb,update_v_w,update_v_hb,update_v_vb,vstate,visibles]

    def reconstruct(self,visibles):
        h_prob = self.actV2H(visibles)
        # reconstruct phase
        for i in range(self.CD_k):
            v_recon = self.actH2V(h_prob)
            h_prob = self.actV2H(v_recon)
        
        recon_error = self.error(visibles,v_recon)
        return v_recon,recon_error

    def error(self,vis,vis_recon):
        if self.cost_fun.lower() == 'mse':
            # mean squared error (linear regression)
            loss = tf.reduce_mean(tf.square(vis - vis_recon))
        elif self.cost_fun.lower() == 'expll':
            # exponential log likelihood (poisson regression)
            loss = tf.reduce_mean(vis_recon - vis*tf.log(vis_recon))
        elif self.cost_fun.lower() == 'xent':
            # cross entropy error (binary classification/Logistic regression)
            loss = -tf.reduce_mean(vis*tf.log(vis_recon)+(1-vis)*tf.log(1-vis_recon+1e-5))
        elif self.cost_fun.lower() == 'mcxent':
            # multi-class (>2) cross entropy (classification) used by softmax
            loss = -tf.reduce_mean(vis*tf.log(vis_recon))
        else:
            raise TypeError("Error type of cost function, should be mse(*)/expll/xent/mcxent")
        return loss


    def pretrain(self,data_x,data_test_x,batch_size=1,n_epoches=1,data_y=None):
        # return errors in training phase
        assert n_epoches > 0 and batch_size > 0
        # define the TF variables
        n_data = data_x.shape[0]
        x_in = tf.placeholder(tf.float32,shape=[None,self.n_visible])
        dropout_keep_prob = tf.placeholder(tf.float32)

##        if data_y is not None:
##            n_out = data_y.shape[1]
##            y_out = tf.placeholder(tf.float32,shape=[None,n_out])
            
        rbm_pretrain = self.rbm_train(x_in, dropout_keep_prob)
        x_re,x_loss = self.reconstruct(x_in)
        
        n_batches = n_data // batch_size
        
        if n_batches == 0:
            n_batches = 1
        
        # deep copy
        data_x_cpy = data_x.copy()
        inds = np.arange(n_data)
        
        # whether or not plot
        if self.plot is True:
            plt.ion() # start the interactive mode of plot
            plt.figure(1)
            
        
        errs = []
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            mean_cost = []
            for epoch in range(n_epoches):
                # shuffle
                np.random.shuffle(inds)
                mean_cost = []
                for b in range(n_batches):
                    batch_x = extractminibatch(inds,b,batch_size,data_x_cpy)
                    exe = sess.run(rbm_pretrain,feed_dict = {x_in:batch_x, dropout_keep_prob:0.9})
##                    if epoch == n_epoches-1:
##                        print("visible : " , exe[-1][0])
##                        print("reconstru : " ,exe[-2][0])
                    cost = sess.run(x_loss,feed_dict={x_in:batch_x})
                    mean_cost.append(cost)
                errs.append(np.mean(mean_cost))
#                print('Epoch %d Cost %g' % (epoch, np.mean(mean_cost)))
                print('Epoch %d Cost %g' % (epoch, errs[-1]))
                
                # plot ? 
                if plt.fignum_exists(1):
                    plt.plot(range(epoch+1),errs,'-r')
            self.train_error = errs


            # test
            self.test = sess.run(x_re,feed_dict = {x_in:data_test_x})
            return errs
    
    def free_energy(self,visibles):
        # ref:http://deeplearning.net/tutorial/rbm.html
        first_term = tf.matmul(visibles,tf.reshape(self.visible_bias,
                                                   [tf.shape(self.visible_bias)[0],1]))
        second_term = tf.reduce_sum(tf.log(1+tf.exp(self.hidden_bias+tf.matmul(visibles,self.w))),axis=1)
        
        return -first_term - second_term
    
