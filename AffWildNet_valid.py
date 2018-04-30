import tensorflow as tf
import numpy as np


class CNN_GRU_3RNN(object):

    def __init__(self,seq_length,batch_size,h_units=128):
        self.params = None
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.h_units = h_units
        self.vars = []
        self.layers = []
        self.names = []  
        # (1): nn.SpatialConvolutionMM(3 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','1',3,3,3,64))
        # (3): nn.SpatialConvolutionMM(64 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','3',3,3,64,64))
        # (5): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (6): nn.SpatialConvolutionMM(64 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','6',3,3,64,128))
        # (8): nn.SpatialConvolutionMM(128 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','8',3,3,128,128))
        # (10): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (11): nn.SpatialConvolutionMM(128 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','11',3,3,128,256))
        # (13): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','13',3,3,256,256))
        # (15): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','15',3,3,256,256))
        # (17): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (18): nn.SpatialConvolutionMM(256 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','18',3,3,256,512))
        # (20): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','20',3,3,512,512))
        # (22): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','22',3,3,512,512))
        # (24): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (25): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','25',3,3,512,512))
        # (27): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','27',3,3,512,512))
        # (29): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','29',3,3,512,512))
        # (31): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (32): nn.View
        # (33): nn.Linear(25088 -> 4096)
        # True = if we have relu activation function else if False = linear activation function
        self.layers.append(('linear','33',4096,True))
        self.layers.append(('rnn_fc','40i',h_units,False)) 
        self.layers.append(('rnn_last_conv','40ii',h_units,False))
        self.layers.append(('rnn_max_pool','40iii',2,False))



    def get_unique_name_(self, prefix):
        id = sum(t.startswith(prefix) for t,_,_ in self.vars)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var,layer):
        self.vars.append((name, var,layer))

    def get_output(self):
        return self.vars[-1][1]

    def make_var(self, name, shape,trainable):
        return tf.get_variable(name, shape,trainable=trainable)

    def setup(self,image_batch):
        self.vars.append(('input',image_batch,['input']))
        for layer in self.layers:
            name = self.get_unique_name_(layer[0])
            if layer[0] == 'conv':
                with tf.variable_scope(name) as scope:
                    h, w, c_i, c_o = layer[2],layer[3],layer[4],layer[5]
                    kernel = self.make_var('weights', shape=[h, w, c_i, c_o],trainable=True)
                    conv = tf.nn.conv2d(self.get_output(), kernel, [1]*4, padding='SAME')
                    biases = self.make_var('biases', [c_o],trainable=True)
                    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                    relu = tf.nn.relu(bias, name=scope.name)
                    self.add_(name, relu,layer)
            elif layer[0] == 'pool':
                size,size,stride,stride = layer[1],layer[2],layer[3],layer[4]
                pool = tf.nn.max_pool(self.get_output(),
                                      ksize=[1, size, size, 1],
                                      strides=[1, stride, stride, 1],
                                      padding='SAME',
                                      name=name)
                self.add_(name, pool,layer)
            elif layer[0] == 'linear':

                num_out = layer[2]
                relu = layer[3]

                with tf.variable_scope(name) as scope:
                    input = self.get_output()
                    input_shape = input.get_shape()
                    if input_shape.ndims==4:
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(input, [self.batch_size*self.seq_length, dim])
                    else:
                        feed_in, dim = (input, int(input_shape[-1]))
                    weights = self.make_var('weights', shape=[dim, num_out],trainable=True)
                    biases = self.make_var('biases', [num_out],trainable=True)
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    fc = op(feed_in, weights, biases, name=scope.name)
                    self.add_(name, fc,layer)

            elif layer[0] == 'rnn_fc':
                num_out = layer[2]
                relu = layer[3]

                with tf.variable_scope(name) as scope:
                    input = self.get_output()
                    input_shape = input.get_shape()
                    if input_shape.ndims==4:
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(input, [self.batch_size*self.seq_length, dim])
                    else:
                        feed_in, dim = (input, int(input_shape[-1]))
                    feed_in = tf.reshape(feed_in,[self.batch_size,self.seq_length,-1])
                    cell =  tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.h_units) for _ in range(2)])
                    outputs, _ = tf.nn.dynamic_rnn(cell, feed_in, dtype=tf.float32)
                    outputs = tf.reshape(outputs, (self.batch_size * self.seq_length, self.h_units))
                    self.add_(name, outputs,layer)

            elif layer[0] == 'rnn_last_conv':
                num_out = layer[2]
                relu = layer[3]

                with tf.variable_scope(name) as scope:
                    last_conv = self.vars[-4][1]
                    feed_in = tf.reshape(last_conv,[self.batch_size,self.seq_length,-1])
                    cell =  tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.h_units) for _ in range(2)])
                    outputs, _ = tf.nn.dynamic_rnn(cell, feed_in, dtype=tf.float32)
                    outputs = tf.reshape(outputs, (self.batch_size * self.seq_length, self.h_units))
                    self.add_(name, outputs,layer)


            elif layer[0] == 'rnn_max_pool':
                num_out = layer[2]
                relu = layer[3]

                with tf.variable_scope(name) as scope:
                    last_pool = self.vars[-4][1]

                    feed_in = tf.reshape(last_pool,[self.batch_size,self.seq_length,-1])
                    cell =  tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.h_units) for _ in range(2)])
                    outputs, _ = tf.nn.dynamic_rnn(cell, feed_in, dtype=tf.float32)
                    outputs = tf.reshape(outputs, (self.batch_size * self.seq_length, self.h_units))

                    rnn_fc = self.vars[-2][1]
                    rnn_last_conv = self.vars[-1][1]
                    total_out = tf.concat([rnn_last_conv,outputs,rnn_fc],1)
                    
                    weights = self.make_var('weights', shape=[3*self.h_units, num_out],trainable=True)
                    biases = self.make_var('biases', [num_out],trainable=True)
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    predictions = op(total_out, weights, biases, name=scope.name)
                    self.add_(name, predictions,layer)



class CNN_GRU_1RNN(object):

    def __init__(self,seq_length,batch_size,h_units=256):
        self.params = None
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.h_units = h_units
        self.vars = []
        self.layers = []
        self.names = []  
        # (1): nn.SpatialConvolutionMM(3 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','1',3,3,3,64))
        # (3): nn.SpatialConvolutionMM(64 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','3',3,3,64,64))
        # (5): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (6): nn.SpatialConvolutionMM(64 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','6',3,3,64,128))
        # (8): nn.SpatialConvolutionMM(128 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','8',3,3,128,128))
        # (10): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (11): nn.SpatialConvolutionMM(128 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','11',3,3,128,256))
        # (13): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','13',3,3,256,256))
        # (15): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','15',3,3,256,256))
        # (17): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (18): nn.SpatialConvolutionMM(256 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','18',3,3,256,512))
        # (20): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','20',3,3,512,512))
        # (22): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','22',3,3,512,512))
        # (24): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (25): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','25',3,3,512,512))
        # (27): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','27',3,3,512,512))
        # (29): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','29',3,3,512,512))
        # (31): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (32): nn.View
        # (33): nn.Linear(25088 -> 4096)
        # True = if we have relu activation function else if False = linear activation function
        self.layers.append(('linear','33',4096,True))
        self.layers.append(('rnn_out_fc_max_pool','40',2,False))



    def get_unique_name_(self, prefix):
        id = sum(t.startswith(prefix) for t,_,_ in self.vars)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var,layer):
        self.vars.append((name, var,layer))

    def get_output(self):
        return self.vars[-1][1]

    def make_var(self, name, shape,trainable):
        return tf.get_variable(name, shape,trainable=trainable)

    def setup(self,image_batch):
        self.vars.append(('input',image_batch,['input']))
        for layer in self.layers:
            name = self.get_unique_name_(layer[0])
            if layer[0] == 'conv':
                with tf.variable_scope(name) as scope:
                    h, w, c_i, c_o = layer[2],layer[3],layer[4],layer[5]
                    kernel = self.make_var('weights', shape=[h, w, c_i, c_o],trainable=True)
                    conv = tf.nn.conv2d(self.get_output(), kernel, [1]*4, padding='SAME')
                    biases = self.make_var('biases', [c_o],trainable=True)
                    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                    relu = tf.nn.relu(bias, name=scope.name)
                    self.add_(name, relu,layer)
            elif layer[0] == 'pool':
                size,size,stride,stride = layer[1],layer[2],layer[3],layer[4]
                pool = tf.nn.max_pool(self.get_output(),
                                      ksize=[1, size, size, 1],
                                      strides=[1, stride, stride, 1],
                                      padding='SAME',
                                      name=name)
                self.add_(name, pool,layer)
            elif layer[0] == 'linear':

                num_out = layer[2]
                relu = layer[3]

                with tf.variable_scope(name) as scope:
                    input = self.get_output()
                    input_shape = input.get_shape()
                    if input_shape.ndims==4:
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(input, [self.batch_size*self.seq_length, dim])
                    else:
                        feed_in, dim = (input, int(input_shape[-1]))
                    weights = self.make_var('weights', shape=[dim, num_out],trainable=True)
                    biases = self.make_var('biases', [num_out],trainable=True)
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    fc = op(feed_in, weights, biases, name=scope.name)
                    self.add_(name, fc,layer)
                    
            elif layer[0] == 'rnn_out_fc_max_pool':
                num_out = layer[2]
                relu = layer[3]

                with tf.variable_scope(name) as scope:
                    last_pool = self.vars[-2][1]
                    last_pool=tf.reshape(last_pool,[self.batch_size*self.seq_length,-1])
                    fc = self.vars[-1][1]
                    feed_in = tf.concat([fc,last_pool],1)
                  
                    feed_in = tf.reshape(feed_in,[self.batch_size,self.seq_length,-1])
                    cell =  tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.h_units) for _ in range(2)])
                    outputs, _ = tf.nn.dynamic_rnn(cell, feed_in, dtype=tf.float32)
                    outputs = tf.reshape(outputs, (self.batch_size * self.seq_length, self.h_units))
                    
                    weights = self.make_var('weights', shape=[self.h_units, num_out],trainable=True)
                    biases = self.make_var('biases', [num_out],trainable=True)
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    predictions = op(outputs, weights, biases, name=scope.name)
                    self.add_(name, predictions,layer)


